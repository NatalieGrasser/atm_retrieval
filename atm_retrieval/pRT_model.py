import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
from PyAstronomy.pyasl import fastRotBroad, helcorr
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import pickle
import pathlib
from scipy.optimize import nnls

import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    from atm_retrieval.cloud_cond import simple_cdf_MgSiO3,return_XMgSiO3
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
elif getpass.getuser() == "natalie": # when testing from my laptop
    from cloud_cond import simple_cdf_MgSiO3,return_XMgSiO3
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"

class pRT_spectrum:

    def __init__(self,
                 retrieval_object,
                 spectral_resolution=100_000,  
                 contribution=False, # only for plotting atmosphere.contr_em
                 interpolate=True):
        
        self.primary_label=retrieval_object.primary_label
        if self.primary_label==False:
            self.primary_wave=retrieval_object.primary_wave
            self.primary_flux=retrieval_object.primary_flux
            self.data_flux = retrieval_object.data_flux
            self.data_err = retrieval_object.data_err

        self.params=retrieval_object.parameters.params
        self.data_wave=retrieval_object.data_wave
        self.target=retrieval_object.target
        self.coords = SkyCoord(ra=self.target.ra, dec=self.target.dec, frame='icrs')
        self.species=retrieval_object.species
        self.spectral_resolution=spectral_resolution
        self.chemistry=retrieval_object.chemistry
        self.atmosphere_objects=retrieval_object.atmosphere_objects
        self.lbl_opacity_sampling=retrieval_object.lbl_opacity_sampling
        self.interpolate=interpolate

        self.n_atm_layers=retrieval_object.n_atm_layers
        self.pressure = retrieval_object.pressure
        self.PT_type=retrieval_object.PT_type
        self.temperature = self.make_pt() #P-T profile

        self.give_absorption_opacity=None
        self.int_opa_cloud = np.zeros_like(self.pressure)
        self.gravity = 10**self.params['log_g'] 
        self.contribution=contribution
        self.cloud_mode=retrieval_object.cloud_mode

        # add_cloud_scat_as_abs, sigma_lnorm, fsed, Kzz only relevant for physical clouds (e.g. MgSiO3)
        self.sigma_lnorm=None
        self.Kzz=None
        self.fsed=None 
        self.add_cloud_scat_as_abs=False

        if self.chemistry=='freechem': # use free chemistry with defined VMRs
            self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species,self.params)
            self.MMW = self.mass_fractions['MMW']

        if self.chemistry in ['equchem','quequchem']: # use equilibium chemistry
            self.abunds = self.abundances(self.pressure,self.temperature,self.params['Fe/H'],self.params['C/O'])
            self.mass_fractions = self.get_abundance_dict(self.species,self.abunds)
            # update mass_fractions with isotopologue ratios
            self.mass_fractions = self.get_isotope_mass_fractions(self.species,self.mass_fractions,self.params) 
            self.MMW = self.abunds['MMW']
            self.VMRs = self.get_VMRs(self.mass_fractions)

        self.spectrum_orders=[]
        self.n_orders=retrieval_object.n_orders

    def abundances(self,press, temp, feh, C_O):
        COs = np.ones_like(press)*C_O
        fehs = np.ones_like(press)*feh
        mass_fractions = interpol_abundances(COs,fehs,temp,press)
        if self.chemistry=='quequchem':
            for species in ['CO','H2O','CH4']:
                Pqu=10**self.params['log_Pqu_CO_CH4'] # is in log
                idx=find_nearest(self.pressure,Pqu)
                quenched_fraction=mass_fractions[species][idx]
                mass_fractions[species][:idx]=quenched_fraction
            for species in ['NH3','HCN']:    
                Pqu=10**self.params[f'log_Pqu_{species}'] # is in log
                idx=find_nearest(self.pressure,Pqu)
                quenched_fraction=mass_fractions[species][idx]
                mass_fractions[species][:idx]=quenched_fraction
        return mass_fractions

    def get_VMRs(self,mass_fractions):
        species_info = pd.read_csv(os.path.join('species_info.csv'))
        VMR_dict={}
        MMW=self.MMW
        for pRT_name in mass_fractions.keys():
            mass=species_info.loc[species_info["pRT_name"]==pRT_name]['mass'].values[0]
            name=species_info.loc[species_info["pRT_name"]==pRT_name]['name'].values[0]
            VMR_dict[name]=mass_fractions[pRT_name]*MMW/mass
        return VMR_dict

    def get_abundance_dict(self,species,abunds): # does not inlcude isotopes
        mass_fractions = {}
        for specie in species:
            if specie in ['H2O_main_iso','H2O_pokazatel_main_iso']:
                mass_fractions[specie] = abunds['H2O']
            elif specie=='CO_main_iso':
                mass_fractions[specie] = abunds['CO']
            elif specie in ['CH4_main_iso','CH4_hargreaves_main_iso']:
                mass_fractions[specie] = abunds['CH4']
            elif specie=='HCN_main_iso':
                mass_fractions[specie] = abunds['HCN']
            elif specie=='NH3_coles_main_iso':
                mass_fractions[specie] = abunds['NH3']
            elif specie=='HF_main_iso':
                mass_fractions[specie] = 1e-12 #abunds['HF'] not in pRT chem equ table
                #species_info = pd.read_csv(os.path.join('species_info.csv'))
                #mass=species_info.loc[species_info["name"]=='HF']['mass'].values[0]
                #mass_fractions[specie] = 1e-12*np.ones(self.n_atm_layers)*mass/self.abunds['MMW'] #abunds['HF'] not in pRT chem equ table, include here
                #mass_fractions[specie] = self.params['log_HF']*mass/self.abunds['MMW'] # add HF manually, convert VMR
            elif specie=='H2S_ExoMol_main_iso':
                mass_fractions[specie] = abunds['H2S']
            elif specie=='OH_main_iso':
                mass_fractions[specie] = 1e-12*np.ones(self.n_atm_layers) #abunds['OH'] not in pRT chem equ table
            elif specie=='CO2_main_iso':
                mass_fractions[specie] = abunds['CO2']
        mass_fractions['H2'] = abunds['H2']
        mass_fractions['He'] = abunds['He']
        return mass_fractions
    
    def read_species_info(self,species,info_key):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if info_key == 'pRT_name':
            return species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return species_info.loc[species,'Hill_notation']
        if info_key == 'mass':
            return species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return species_info.loc[species,info_key]
        if info_key == 'c' or info_key == 'color':
            return species_info.loc[species,'color']
        if info_key == 'label':
            return species_info.loc[species,'mathtext_name']
    
    def get_isotope_mass_fractions(self,species,mass_fractions,params):
        #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/chemistry.py
        mass_ratio_13CO_12CO = self.read_species_info('13CO','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C18O_C16O = self.read_species_info('C18O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C17O_C16O = self.read_species_info('C17O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_H218O_H2O = self.read_species_info('H2(18)O','mass')/self.read_species_info('H2O','mass')
        self.C13_12_ratio = 10**(-params.get('log_C12_13_ratio',-12))
        self.O18_16_ratio = 10**(-params.get('log_O16_18_ratio',-12))
        self.O17_16_ratio = 10**(-params.get('log_O16_17_ratio',-12))

        for species_i in species:
            if (species_i=='CO_main_iso'): # 12CO mass fraction
                mass_fractions[species_i]=(1-self.C13_12_ratio*mass_ratio_13CO_12CO
                                            -self.O18_16_ratio*mass_ratio_C18O_C16O
                                            -self.O17_16_ratio*mass_ratio_C17O_C16O)*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_36'): # 13CO mass fraction
                mass_fractions[species_i]=self.C13_12_ratio*mass_ratio_13CO_12CO*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_28'): # C18O mass fraction
                mass_fractions[species_i]=self.O18_16_ratio*mass_ratio_C18O_C16O*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_27'): # C17O mass fraction
                mass_fractions[species_i]=self.O17_16_ratio*mass_ratio_C17O_C16O*mass_fractions['CO_main_iso']
                continue
            if (species_i in ['H2O_main_iso','H2O_pokazatel_main_iso']): # H2O mass fraction
                H2O_linelist=species_i
                mass_fractions[species_i]=(1-self.O18_16_ratio*mass_ratio_H218O_H2O)*mass_fractions[species_i]
                continue
            if (species_i=='H2O_181_HotWat78'): # H2_18O mass fraction
                mass_fractions[species_i]=self.O18_16_ratio*mass_ratio_H218O_H2O*mass_fractions[H2O_linelist]
                continue
            
        return mass_fractions
    
    def free_chemistry(self,line_species,params):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

        for species_i in species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if line_species_i in line_species:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[line_species_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
        
        if VMR_wo_H2.any() > 1:
            print('VMR_wo_H2 > 1. Other species are too abundant!')

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for line_species_i in mass_fractions.keys():
            mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        return mass_fractions, CO, FeH

    def gray_cloud_opacity(self): # like in deRegt+2024
        def give_opacity(wave_micron=self.wave_micron,pressure=self.pressure):
            opa_gray_cloud = np.zeros((len(self.wave_micron),len(self.pressure))) # gray cloud = independent of wavelength
            opa_gray_cloud[:,self.pressure>10**(self.params['log_P_base_gray'])] = 0 # [bar] constant below cloud base
            #opa_gray_cloud[:,self.pressure>10**(self.params['log_P_base_gray'])] =10**(self.params['log_opa_base_gray'])
            # Opacity decreases with power-law above the base
            above_clouds = (self.pressure<=10**(self.params['log_P_base_gray']))
            opa_gray_cloud[:,above_clouds]=(10**(self.params['log_opa_base_gray']))*(self.pressure[above_clouds]/10**(self.params['log_P_base_gray']))**self.params['fsed_gray']
            if self.params.get('cloud_slope') is not None:
                opa_gray_cloud *= (self.wave_micron[:,None]/1)**self.params['cloud_slope']
            return opa_gray_cloud
        return give_opacity
    
    def make_spectrum(self):

        spectrum_orders=[]
        self.wlshift_orders=[]
        waves_orders=[]
        self.contr_em_orders=[]
        self.phi_components=np.full(shape=(7,3,2),fill_value=np.nan)
        self.secondary_flux=np.full(shape=(7,3,2048),fill_value=np.nan)
        for order in range(self.n_orders):
            atmosphere=self.atmosphere_objects[order]

            # MgSiO3 cloud model like in Sam's code
            if self.cloud_mode == 'MgSiO3':
                if self.chemistry=='freechem':
                    co=self.CO
                    feh=self.FeH
                if self.chemistry in ['equchem','quequchem']:
                    feh=self.params['Fe/H']
                    co=self.params['C/O']
                P_base_MgSiO3 = simple_cdf_MgSiO3(self.pressure,self.temperature,feh,co,np.nanmean(self.MMW))
                above_clouds = (self.pressure<=P_base_MgSiO3) # mask pressure above cloud deck
                eq_MgSiO3 = return_XMgSiO3(feh, co)
                self.mass_fractions['MgSiO3(c)'] = np.zeros_like(self.temperature)
                condition=eq_MgSiO3*(self.pressure[above_clouds]/P_base_MgSiO3)**self.params['fsed']
                self.mass_fractions['MgSiO3(c)'][above_clouds]=condition
                self.sigma_lnorm=self.params['sigma_lnorm']
                self.Kzz = 10**self.params['log_Kzz']*np.ones_like(self.pressure) 
                self.fsed=self.params['fsed']
                self.add_cloud_scat_as_abs=True

            elif self.cloud_mode == 'gray': # Gray cloud opacity
                self.wave_micron = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
                self.give_absorption_opacity=self.gray_cloud_opacity() # fsed_gray only needed here, not in calc_flux

            atmosphere.calc_flux(self.temperature,
                            self.mass_fractions,
                            self.gravity,
                            self.MMW,
                            Kzz=self.Kzz, # only for MgSiO3 clouds
                            fsed = self.fsed, # only for MgSiO3 clouds 
                            sigma_lnorm = self.sigma_lnorm, # only for MgSiO3 clouds
                            add_cloud_scat_as_abs=self.add_cloud_scat_as_abs, # only for MgSiO3 clouds
                            contribution =self.contribution,
                            give_absorption_opacity=self.give_absorption_opacity)

            wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
            flux=atmosphere.flux
            #flux = atmosphere.flux/np.nanmean(atmosphere.flux)

            # RV+bary shifting and rotational broadening
            v_bary, _ = helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, # of Cerro Paranal
                            ra2000=self.coords.ra.value,dec2000=self.coords.dec.value,jd=self.target.JD) # https://ssd.jpl.nasa.gov/tools/jdc/#/cd
            wl_shifted= wl*(1.0+(self.params['rv']-v_bary)/const.c.to('km/s').value)
            self.wlshift_orders.append(wl_shifted)
            waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
            spec = np.interp(waves_even, wl_shifted, flux)
            spec = fastRotBroad(waves_even, spec, self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
            spec = self.convolve_to_resolution(waves_even, spec, self.spectral_resolution)

            #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/spectrum.py#L289
            self.resolution = int(1e6/self.lbl_opacity_sampling)
            flux=self.instr_broadening(waves_even*1e3,spec,out_res=self.resolution,in_res=500000)

            # Interpolate/rebin onto the data's wavelength grid
            # should not be done when making spectrum for cross-corr, or wl padding will be cut off
            if self.interpolate==True:
                ref_wave = self.data_wave[order].flatten() # [nm]
                flux = np.interp(ref_wave, waves_even*1e3, flux) # pRT wavelengths from microns to nm

                # reshape to (detectors,pixels) so that we can store as shape (orders,detectors,pixels)
                flux=flux.reshape(self.data_wave.shape[1],self.data_wave.shape[2])

            if self.primary_label==False: # should have same wavelengths
                for det in range(3):
                    #nans = np.isnan(self.primary_flux[order][det]) | np.isnan(self.data_flux[order][det])
                    nonans = np.isfinite(self.primary_flux[order][det]) & np.isfinite(self.data_flux[order][det]) & np.isfinite(self.data_err[order][det])
                    #if nans.all()==True: # skip empty
                    if np.sum(nonans)==0:
                        flux[det]/=np.nanmedian(flux[det])
                        continue
                    #M = np.vstack([self.primary_flux[order][det][~nans], flux[det][~nans]]).T # model matrix, shape (2, N)
                    #d = self.data_flux[order][det][~nans]  # prepare data
                    #var = self.data_err[order][det][~nans]**2
                    M = np.vstack([self.primary_flux[order][det][nonans], flux[det][nonans]]).T # model matrix, shape (2, N)
                    d = self.data_flux[order][det][nonans]  # prepare data
                    var = self.data_err[order][det][nonans]**2
                    inv_cov = np.diag(1/var) # inverse of covariance matrix
                    
                    # set up equation to solve (see Ruffio+2019 Appendix A)
                    lhs = M.T @ inv_cov @ M # left-hand side
                    rhs = M.T @ inv_cov @ d # right-hand side
                    phi_comp, _ = nnls(lhs, rhs)
                    #self.phi_components.append(phi_comp)
                    self.phi_components[order,det]=phi_comp
                    self.secondary_flux[order,det]=np.copy(flux[det])
                    flx = phi_comp[1]*flux[det] + phi_comp[0]*self.primary_flux[order][det]

                    #if getpass.getuser() == "natalie": # when testing from my laptop
                    if False:
                        print(f'Linear parameters: {phi_comp}')
                        plt.plot(self.data_wave[order][det], self.data_flux[order][det],c='k', label='data')
                        plt.plot(self.data_wave[order][det], phi_comp[0]*self.primary_flux[order][det], label='A')
                        plt.plot(self.data_wave[order][det], phi_comp[1]*flux[det], label='B')
                        plt.plot(self.data_wave[order][det], flx, label='A+B',linestyle='dotted',c='limegreen')
                        plt.legend()
                        plt.show()
                    
                    #flx/=np.median(flx)
                    flux[det]=flx

            spectrum_orders.append(flux)
            waves_orders.append(waves_even*1e3) # from um to nm

            if self.contribution==True:
                contr_em = atmosphere.contr_em # emission contribution
                summed_contr = np.nansum(contr_em,axis=1) # sum over all wavelengths
                self.contr_em_orders.append(summed_contr)

        if False:
            for order in range(7):
                for det in range(3):
                    phi_comp=self.phi_components[order,det]
                    print(f'Linear parameters: {phi_comp}')
                    plt.plot(self.data_wave[order][det], self.data_flux[order][det],c='k', label='data')
                    plt.plot(self.data_wave[order][det], phi_comp[0]*self.primary_flux[order][det], label='A')
                    plt.plot(self.data_wave[order][det], phi_comp[1]*self.secondary_flux[order][det], label='B')
                    plt.plot(self.data_wave[order][det], spectrum_orders[order][det], label='A+B',linestyle='dotted',c='limegreen')
                    plt.legend()
                    plt.show()
            
        if self.interpolate==False:
            get_median=np.array([])
            for order in range(7): # append value by value because not all the same size
                get_median=np.append(get_median,spectrum_orders[order]) 
            spectrum_orders=np.array(spectrum_orders,dtype=object)
            spectrum_orders/=np.nanmedian(get_median) # orders not same size, np.median didn't work otherwise
            return spectrum_orders, waves_orders
        else:
            spectrum_orders=np.array(spectrum_orders)
            spectrum_orders/=np.nanmedian(spectrum_orders) # normalize in same way as data spectrum
            return spectrum_orders
            
    def make_pt(self,**kwargs): 

        if self.PT_type=='PTknot': # retrieve temperature knots
            self.T_knots = np.array([self.params['T4'],self.params['T3'],self.params['T2'],self.params['T1'],self.params['T0']])
            self.log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=len(self.T_knots))
            sort = np.argsort(self.log_P_knots)
            self.temperature = CubicSpline(self.log_P_knots[sort],self.T_knots[sort])(np.log10(self.pressure))
        
        if self.PT_type=='PTgrad':
            self.log_P_knots = np.linspace(np.log10(np.min(self.pressure)),
                                           np.log10(np.max(self.pressure)),num=5) # 5 gradient values
            
            if 'dlnT_dlnP_knots' not in kwargs:
                self.dlnT_dlnP_knots=[]
                for i in range(5):
                    self.dlnT_dlnP_knots.append(self.params[f'dlnT_dlnP_{i}'])
            elif 'dlnT_dlnP_knots' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                self.dlnT_dlnP_knots=kwargs.get('dlnT_dlnP_knots')

            # interpolate over dlnT/dlnP gradients
            interp_func = interp1d(self.log_P_knots,self.dlnT_dlnP_knots,kind='quadratic') # for the other 50 atm layers
            dlnT_dlnP = interp_func(np.log10(self.pressure))[::-1] # reverse order, start at bottom of atm

            if 'T_base' not in kwargs:
                T_base = self.params['T0'] # T0 is free param, at bottom of atmosphere
            elif 'T_base' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                T_base=kwargs.get('T_base')

            ln_P = np.log(self.pressure)[::-1]
            temperature = [T_base, ]

            # calc temperatures relative to base pressure, from bottom to top of atmosphere
            for i, ln_P_up_i in enumerate(ln_P[1:]): # start after base, T at base already defined
                ln_P_low_i = ln_P[i]
                ln_T_low_i = np.log(temperature[-1])
                # compute temperatures based on gradient
                ln_T_up_i = ln_T_low_i + (ln_P_up_i - ln_P_low_i)*dlnT_dlnP[i+1]
                temperature.append(np.exp(ln_T_up_i))
            self.temperature = temperature[::-1] # reverse order, pRT reads temps from top to bottom of atm
        
        return self.temperature

    def convolve_to_resolution(self, in_wlen, in_flux, out_res, in_res=None):
        from scipy.ndimage import gaussian_filter
        if isinstance(in_wlen, u.Quantity):
            in_wlen = in_wlen.to(u.nm).value
        if in_res is None:
            in_res = np.mean((in_wlen[:-1]/np.diff(in_wlen)))
        # delta lambda of resolution element is FWHM of the LSF's standard deviation:
        sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))
        spacing = np.mean(2.*np.diff(in_wlen)/(in_wlen[1:]+in_wlen[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF/spacing
        out_flux = np.tile(np.nan, in_flux.shape)
        nans = np.isnan(in_flux)
        out_flux[~nans] = gaussian_filter(in_flux[~nans], sigma = sigma_LSF_gauss_filter, mode = 'reflect')
 
        return out_flux

    
    def instr_broadening(self, wave, flux, out_res=1e6, in_res=1e6):

        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2-1/in_res**2)/(2*np.sqrt(2*np.log(2)))
        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter,mode='nearest')

        return flux_LSF
    
    def make_spectrum_continuous(self,ref_wave): # just for plotting, not needed for retrieval

        file=pathlib.Path(f'atmosphere_objects_continuous.pickle')
        if file.exists():
            with open(file,'rb') as file:
                atmosphere=pickle.load(file)

        if self.cloud_mode == 'gray': # Gray cloud opacity
            self.wave_micron = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
            self.give_absorption_opacity=self.gray_cloud_opacity() # fsed_gray only needed here, not in calc_flux

        atmosphere.calc_flux(self.temperature,
                        self.mass_fractions,
                        self.gravity,
                        self.MMW,
                        Kzz=self.Kzz, # only for MgSiO3 clouds
                        fsed = self.fsed, # only for MgSiO3 clouds 
                        sigma_lnorm = self.sigma_lnorm, # only for MgSiO3 clouds
                        add_cloud_scat_as_abs=self.add_cloud_scat_as_abs, # only for MgSiO3 clouds
                        contribution =self.contribution,
                        give_absorption_opacity=self.give_absorption_opacity)

        wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
        flux=atmosphere.flux

        # RV+bary shifting and rotational broadening
        v_bary, _ = helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, # of Cerro Paranal
                        ra2000=self.coords.ra.value,dec2000=self.coords.dec.value,jd=self.target.JD) # https://ssd.jpl.nasa.gov/tools/jdc/#/cd
        wl_shifted= wl*(1.0+(self.params['rv']-v_bary)/const.c.to('km/s').value)
        waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
        spec = np.interp(waves_even, wl_shifted, flux)
        spec = fastRotBroad(waves_even, spec, self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
        spec = self.convolve_to_resolution(waves_even, spec, self.spectral_resolution)
        flux = np.interp(ref_wave, waves_even*1e3, flux) # pRT wavelengths from microns to nm
        flux/=np.nanmedian(flux)

        return flux

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx