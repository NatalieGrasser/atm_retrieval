import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import pathlib
import astropy.constants as const
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import numpy.ma as ma
from scipy.ndimage import convolve1d
from petitRADTRANS import nat_cst as nc
from scipy import constants as sc
import pathlib
from scipy import spatial
import warnings

warnings.simplefilter("error", RuntimeWarning)  # Convert warnings to exceptions
import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    path_tables = '/net/lem/data2/regt/fastchem_tables'
elif getpass.getuser() == "natalie": # when testing from my laptop
    path_tables = '/home/natalie/fastchem_tables'

def scale_between(ymin,ymax,arr):
    try:
        scale=(ymax-ymin)/(np.nanmax(arr)-np.nanmin(arr))
    except RuntimeWarning:
        scale = (ymax-ymin)
    scaled_arr=scale*(arr-np.nanmin(arr))+ymin
    return scaled_arr

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def blackbody(wl_nm,temp,norm=True):
    lamb = wl_nm*1e-7 # wavelength array in cm
    freq = nc.c / lamb # Convert to frequencies
    planck = nc.b(temp, freq) # Calculate Planck function at given temperature (K)
    if norm:
        planck=planck/np.mean(planck)
    return planck

def rem_cont(wave,spec,poly_order=3):
    spec=spec.flatten()
    wave=wave.flatten()
    flux_contrem = np.full(spec.shape,np.nan)
    continuum_model = np.poly1d(np.polyfit(wave,spec,poly_order))
    continuum = continuum_model(wave)
    flux_contrem = spec/continuum      
    return flux_contrem

def get_ratios(retr_obj,equ_too=False): # in case not all ratios are in retrieval
    if retr_obj.chemistry in ['equchem','quequchem','flexequ']:
        if retr_obj.chemistry in ['equchem','quequchem']:
            ratios_default = ['C/O','Fe/H']
        elif retr_obj.chemistry=='flexequ':
            ratios_default = ['C/O','C/H']
        check_ratios = ['log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_H2O16_18_ratio']
        for r in check_ratios:
            if r in retr_obj.parameters.param_keys:
                ratios_default.append(r)
    elif retr_obj.chemistry in ['freechem','varchem']:
        suffix='_0' if retr_obj.chemistry=='varchem' else ''
        ratios_default = ['C/O','C/H']
        ratios_default_equ = ['C/O','Fe/H']
        check_ratios = ['log_12CO/13CO','log_12CO/C17O','log_12CO/C18O','log_H2O/H2(18)O']
        ratios_equ = ['log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_H2O16_18_ratio']
        check_molec = [f'13CO{suffix}',f'C17O{suffix}',f'C18O{suffix}',f'H2(18)O{suffix}']
        for m,r,e in zip(check_molec,check_ratios,ratios_equ):
            if f'log_{m}' in retr_obj.parameters.param_keys:
                ratios_default.append(r)
                ratios_default_equ.append(e)
    if equ_too==False:
        return ratios_default
    else:
        return ratios_default, ratios_default_equ

# cross-correlate residuals with spectrum that contains selected species at equilibrium
def CCF_residuals(retr_obj,ccf_species=[],use_equ=True,noiserange=100): # can only be run after evaluate()

    from retrieval import Retrieval
    from parameters import Parameters
    from pRT_model import pRT_spectrum
    import figures as figs

    ccf_dict={}
    ccf_acf_dict={} # save cross-correlations and auto-correlations
    # function only for CRIRES spectra anyway
    crires_shape = (retr_obj.n_orders,retr_obj.n_dets,retr_obj.n_pixels)
    to_reshape= [retr_obj.data_flux, retr_obj.data_err, retr_obj.data_wave, retr_obj.mask_isfinite, retr_obj.model_flux]
    data_flux, data_err, data_wave, mask_isfinite, model_flux = [var.reshape(crires_shape) for var in to_reshape]
    Cov = retr_obj.Cov.reshape(crires_shape[:-1])

    if ccf_species==[]:
        leave_out = ['13CH4','H2(17)O','H2(18)O','C17O','C18O','13CO','H2','He']
        for species_i in list(retr_obj.species_info.index.values):
            if species_i not in retr_obj.species_names + leave_out:
                ccf_species.append(species_i) # all that aren't retrieved

    if isinstance(ccf_species, list)==False:
        ccf_species=[ccf_species] # if only one, make list so that it works in the for loop
    
    RVs=np.arange(-500,500,1) # km/s
    for ccf_species_i in ccf_species:

        # create template with only selected species at equilibrium abundance
        hill_i = retr_obj.species_info.loc[ccf_species_i,'Hill_notation']
        equ_table = pathlib.Path(f'{path_tables}/{hill_i}.hdf5')
        parameters_spec = retr_obj.params_dict
        if equ_table.exists() and use_equ==True:
            usechem= 'equchem'
            parameters_spec.update({'C/O': retr_obj.params_dict['C/O'],
                            'Fe/H': retr_obj.params_dict['C/H']})
            ratios_free,ratios_equ = get_ratios(retr_obj,equ_too=True)
            for r,e in zip(ratios_free,ratios_equ):
                parameters_spec.update({e: retr_obj.params_dict[r]})
        else:
            usechem= 'freechem'
            for other_spec_i in retr_obj.species_names:
                parameters_spec.pop(f'log_{other_spec_i}', None)
            parameters_spec[f'log_{ccf_species_i}']=-5 # manually set abundance

        parameters_spec = Parameters({}, parameters_spec)
        parameters_spec.param_priors['log_l']=[-3,0]
        retr_spec = Retrieval(target=retr_obj.target,parameters=parameters_spec, 
                                species_names=retr_obj.species_names,Nlive=retr_obj.Nlive,
                                evtol=retr_obj.evtol,chemistry=usechem,
                                PT_type=retr_obj.PT_type,cloud_mode=retr_obj.cloud_mode)
        retr_spec.primary_label=True
        retr_spec.species_names = [ccf_species_i]
        retr_spec.species_pRT, retr_spec.species_hill =retr_spec.get_pRT_hill(retr_spec.species_names)
        retr_spec.atmosphere_objects = retr_spec.get_atmosphere_objects(for_species=ccf_species_i)
        template_fluxes, template_waves =pRT_spectrum(retr_spec,interpolate=False).make_spectrum()
        
        beta=1.0-RVs/const.c.to('km/s').value
        CCF = np.zeros((retr_obj.n_orders,retr_obj.n_dets,len(RVs)))
        ACF = np.zeros((retr_obj.n_orders,retr_obj.n_dets,len(RVs))) # auto-correlation

        for order in range(retr_obj.n_orders):
            for det in range(retr_obj.n_dets):

                if np.isnan(data_flux[order,det]).all():
                    pass # skip empty order/det, CCF and ACF remains 0 

                else:
                    
                    template_flux=template_fluxes[order]
                    template_wl = template_waves[order]
                    template_flux = rem_cont(template_wl,template_flux)
                    
                    wl_data= data_wave[order,det,mask_isfinite[order,det]]
                    fl_data = data_flux[order,det,mask_isfinite[order,det]]-model_flux[order,det,mask_isfinite[order,det]]
                    #plt.plot(template_wl,template_flux,c='tab:blue')
                    #plt.plot(wl_data,fl_data,c='tab:orange')
                    fl_data-=np.nanmean(fl_data)

                    Cov[order,det].get_cholesky() # in case it hasn't been called yet
                    cov_0_data=Cov[order,det].solve(fl_data)                            
                    wl_shift=wl_data[:, np.newaxis]*beta[np.newaxis, :]
                    template_shift=interp1d(template_wl,template_flux)(wl_shift) # interpolate template onto shifted wl
                    #template_shift-= np.nanmedian(template_shift)  
                    template_shift = np.array([template_shift[:,i] - np.nanmedian(template_shift[:,i]) for i in range(template_shift.shape[1])]).T
                    #print(order,det, np.nanmedian(template_shift),np.nanmedian(fl_data))
                    #temptemplate_shift[:,len(RVs)//2]

                    template_rebinned=interp1d(template_wl,template_flux)(wl_data)
                    template_rebinned-=np.nanmedian(template_rebinned)
                    cov_0_temp=Cov[order,det].solve(template_rebinned)

                    CCF[order,det]=(template_shift.T).dot(cov_0_data)
                    ACF[order,det]=(template_shift.T).dot(cov_0_temp)

        CCF_sum=np.sum(np.sum(CCF,axis=0),axis=0) # sum CCF over all orders detectors
        ACF_sum=np.sum(np.sum(ACF,axis=0),axis=0)
        noise=np.std(CCF_sum[np.abs(RVs)>noiserange]) # mask out regions close to expected RV
        if noise==0:
            CCF_norm = np.full(CCF_sum.shape,-1)
            ACF_norm = np.full(CCF_sum.shape,-1)
        else:
            CCF_norm = CCF_sum/noise # get ccf map in S/N units
            ACF_norm = ACF_sum/noise

        SNR=CCF_norm[np.where(RVs==0)[0][0]]
        ccf_dict[f'SNR_{ccf_species_i}']=SNR
        ccf_acf_dict[ccf_species_i]=(CCF_norm,ACF_norm,SNR)
        print(f'{ccf_species_i} S/N =',np.round(SNR,decimals=2))

    retr_obj.ccf_acf_dict = ccf_acf_dict
    suf='_res_equ' if use_equ else '_res'
    figs.CCF_plot_all(retr_obj,ccf_species,noiserange=100,show_ACF=True,suffix=suf)

    # folder created when initializing retrieval object, delete afterwards
    if os.path.isdir(retr_spec.output_dir) and not os.listdir(retr_spec.output_dir):  # Check if folder exists and is empty
        os.rmdir(retr_spec.output_dir)  # Remove empty folder

    return ccf_dict

class PSG_input: # for LIFE retrievals

    def __init__(self,name):
        self.name = name
        self.table = self.create_table()
        self.pressure = self.table['Pressure'].to_numpy()
        self.temperature = self.table['Temperature'].to_numpy()

    def create_table(self): # convert PSG input file into useable table

        with open(f'./LIFE/{self.name}/{self.name}_psg_input.txt', "r") as file:
            lines = file.readlines()

        rows = []
        for line in lines:
            if "<ATMOSPHERE-LAYER-" in line:
                index = line.index(">")
                rows.append(line[index+1:-1]) # remove \n from end of row

        columns = [ "Pressure", "Temperature", "Altitude", "H2", "He", "H2O", "CH4", "C2H6", "CO2", "C2H2", "C2H4", "CO",
                    "H2CO", "NH3", "SO2", "H2S", "SO", "CS2", "OCS", "DMS", "C2H6S2"]

        df = pd.DataFrame([row.split(",") for row in rows])
        df.columns = columns
        df = df.astype(float) # Convert all columns to float
        df = df.iloc[::-1] # reverse order, bc pRT reads temps from top to bottom of atmosphere

        return df

# from DGonzalezPicos/broadpy
class InstrumentalBroadening:
    
    c = const.c.to(u.km/u.s).value
    sqrt8ln2 = np.sqrt(8 * np.log(2))
    
    available_kernels = ['gaussian','gaussian_variable']
    
    def __init__(self, x, y):
        
        self.x = x # units of wavelength
        self.y = y # units of flux (does not matter)
        self.spacing = np.mean(2*np.diff(self.x) / (self.x[1:] + self.x[:-1]))
    
    def __call__(self, res=None, fwhm=None, gamma=None, truncate=4.0, kernel='auto'):
        '''Instrumental broadening
        provide either instrumental resolution lambda/delta_lambda or FWHM in km/s'''
        kernel = self.__read_kernel(res=res, fwhm=fwhm, gamma=gamma) if kernel == 'auto' else kernel
        
        if kernel == 'gaussian':
            fwhm = fwhm if fwhm is not None else (self.c / res)
            _kernel = self.gaussian_kernel(fwhm, truncate)
            
        if kernel == 'gaussian_variable':
            _kernels, lw = self.gaussian_variable_kernel(fwhm, truncate)
            y_pad = np.pad(self.y, (lw, lw), mode='reflect')
            y_matrix = np.lib.stride_tricks.sliding_window_view(y_pad, window_shape=(2 * lw + 1))
            y_lsf = np.einsum('ij, ij->i', _kernels, y_matrix)
            return y_lsf
            
        y_lsf = convolve1d(self.y, _kernel, mode='nearest')
        return y_lsf
    
    @classmethod
    def gaussian_profile(self, x, x0, sigma):
        '''Gaussian function'''
        return np.exp(-0.5 * ((x - x0) / sigma)**2)# / (sigma * np.sqrt(2*np.pi))
    
    def gaussian_kernel(self,fwhm,truncate=4.0,):
        ''' Gaussian kernel
        
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        # Adapted from scipy.ndimage.gaussian_filter1d        
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd + 0.5)
    
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.gaussian_profile(kernel_x, 0, sd)
        kernel /= np.sum(kernel)  # normalize the kernel
        return kernel
    
    def gaussian_variable_kernel(self, fwhm, truncate=4.0):
        ''' Gaussian kernel with variable FWHM
        
        Parameters
        ----------
        fwhm : array
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd.max() + 0.5)
        x = np.arange(-lw, lw + 1)
        
        # Use broadcasting to create a 2D array of Gaussian kernels
        kernels = np.exp(-0.5 * (x[None, :] / sd[:, None]) ** 2)
        kernels /= kernels.sum(axis=1)[:, None]
        return kernels, lw

def fill_nan_nearest(arr): # fill nans with the nearest non-nan
    orig_shape = arr.shape
    arr = arr.flatten()
    nans = np.isnan(arr)
    not_nans = np.where(~nans)[0]
    if not len(not_nans):
        raise ValueError("Array is all NaNs.")
    nearest_index = np.abs(np.subtract.outer(np.arange(len(arr)), not_nans)).argmin(1)
    arr = arr[not_nans[nearest_index]]
    arr = arr.reshape(orig_shape)
    return arr

# get continuum through polynomial fit
def fit_continuum_1D(wl,fl,poly_order=3):
    nans = np.isnan(fl)
    if np.sum(nans)>(len(fl)-10):
        return np.ones_like(fl)
    continuum = np.poly1d(np.polyfit(wl[~nans],fl[~nans],poly_order))(wl)
    return continuum

# fit continuum through upper envelope and remove it
def upper_envelope_remove_continuum(wavelength, flux, min_valid_per_bin=250,
                        percentile=97, spline_smoothing=0.001, oneD=False):
    """
    Estimate continuum using adaptive binning based on non-NaN values.

    Parameters:
    - wavelength: 1D array of wavelengths
    - flux: 1D array of fluxes (can include NaNs)
    - min_valid_per_bin: minimum number of non-NaN points per bin
    - percentile: upper percentile to use for continuum points
    - spline_smoothing: smoothing parameter for the spline

    Returns:
    - continuum: estimated continuum over the full wavelength range
    """

    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)
    if np.isnan(flux).all():
        return flux
    if np.sum(np.isnan(flux))>1450 or oneD==True:
        cont = fit_continuum_1D(wavelength,flux,poly_order=1)
        flux/=cont
        flux/=np.nanpercentile(flux,percentile)
        return flux
    elif np.sum(np.isnan(flux))>1000:
        min_valid_per_bin = 50 #use smaller bins

    n = len(wavelength)
    i = 0

    max_waves, max_fluxes = [], []

    while i < n:
        j = i
        count = 0

        # Grow the bin until it has enough non-NaN values
        while j < n and count < min_valid_per_bin:
            if not (np.isnan(flux[j]) or np.isnan(wavelength[j])):
                count += 1
            j += 1

        if count == 0:
            i = j
            continue

        # Work with the valid bin
        w_bin = wavelength[i:j]
        f_bin = flux[i:j]
        mask = ~np.isnan(w_bin) & ~np.isnan(f_bin)
        w_clean = w_bin[mask]
        f_clean = f_bin[mask]

        if len(f_clean) == 0:
            i = j
            continue

        threshold = np.percentile(f_clean, percentile)
        idx = np.where(f_clean >= threshold)[0][0]

        max_waves.append(w_clean[idx])
        max_fluxes.append(f_clean[idx])

        # Move to next bin (non-overlapping)
        i = j

    if len(max_waves) < 4:
        return flux
        #raise ValueError("Too few continuum points extracted. Try lowering min_valid_per_bin or checking your data.")
    else:
        # Fit spline to the upper envelope
        spline = UnivariateSpline(max_waves, max_fluxes, s=spline_smoothing)
        continuum = spline(wavelength)

    flux/=continuum
    #flux/=np.nanmax(flux)

    return flux

def fft_remove_continuum(flux, lower_cutoff=None):

    if np.isnan(flux).all():
        return flux
        
    if lower_cutoff==None:
        # Start by removing frequencies below ~1/100 to 1/200 of array length
        lower_cutoff = len(flux) // 150  # ≈ 13

    flux = np.asarray(flux)
    n = len(flux)
    
    # can't handle NaNs
    isnan = np.isnan(flux)
    if np.any(isnan):
        flux = np.interp(np.arange(n), np.arange(n)[~isnan], flux[~isnan])

    # FFT and filtering
    fft_flux = np.fft.rfft(flux)
    fft_flux[:lower_cutoff] = 0  # remove low-frequency continuum
    filtered = np.fft.irfft(fft_flux, n=n) + np.ones(n)
    
    return filtered

def generate_skewed_p_nodes(p_min=1e-6, p_max=1e0, n_nodes=7, skew=2.0):
    """
    Generate pressure nodes in log space, skewed toward the bottom (high pressure).
    
    Parameters:
        p_min: float, minimum pressure (top of atmosphere)
        p_max: float, maximum pressure (bottom)
        n_nodes: int, total number of pressure nodes
        skew: float > 0, higher means more concentrated at top (skew > 1),
                        skew < 1 concentrates at bottom.
    
    Returns:
        Array of pressures in bars.
    """
    x = np.linspace(0, 1, n_nodes)
    x_skewed = x**skew  # Skewed toward 1 (bottom) for skew > 1
    log_p = np.log10(p_min) + x_skewed * (np.log10(p_max) - np.log10(p_min))
    return log_p

def planck_lambda_um(T, lam_um):
    """
    Planck function B_lambda in units of W / m² / μm / sr.
    """
    lam_m = lam_um * 1e-6  # μm → m
    c1 = 2 * sc.h * sc.c**2  # first radiation constant
    c2 = sc.h * sc.c / sc.k  # second radiation constant
    exponent = c2 / (lam_m * T)
    B_lambda = (c1 / (lam_m**5)) / (np.exp(exponent) - 1)  # W / m² / m / sr
    return B_lambda * 1e-6  # → W / m² / μm / sr
