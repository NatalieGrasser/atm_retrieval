import numpy as np
import pandas as pd
import pickle
import os
os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"

obj = 'Sorg20X'
CIA_species =  ['H2-H2','H2-He','CH4-CH4','CH4-He','CO2-CH4','CO2-CO2','CO2-H2','CO2-He','H2-CH4','H2O-H2O']
species_names=["H2", "He", "H2O", "CH4", "C2H6", "CO2", "C2H2", "C2H4", "CO",
                "H2CO", "NH3", "SO2", "H2S", "SO", "CS2", "OCS","DMS","C2H6S2"]

def load_PSG_spectrum(resolution):
    #file=np.genfromtxt(f'LIFE/Sorg20X/Sorg20X_PSG{int(resolution)}_spectrum_normalized.txt',skip_header=1,delimiter=' ')
    #if resolution==50:
        #file=np.genfromtxt(f'LIFE/Sorg20X/Sorg20X_spectrum.txt',skip_header=1,delimiter=' ')
    #else:
    file=np.genfromtxt(f'LIFE/Sorg20X/Sorg20X_PSG{int(resolution)}_spectrum.txt',skip_header=1,delimiter=' ')
    wl=file[:,0]
    fl=file[:,1]
    err=file[:,2]
    err/=np.max(fl)
    fl/=np.max(fl)
    return wl,fl,err

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

wl_R1000,fl_R1000,err_R1000 = load_PSG_spectrum(1000)
wl_R50,fl_R50,err_R50 = load_PSG_spectrum(50)

species_info = pd.read_csv('species_info_ck.csv',index_col=0)

class PSG_input: # for LIFE retrievals

    def __init__(self,name):
        self.name = name
        self.table = self.create_table()
        self.pressure = self.table['Pressure'].to_numpy()
        self.temperature = self.table['Temperature'].to_numpy()

    def create_table(self): # convert PSG input file into useable table

        with open(f'./LIFE/Sorg20X/{self.name}_psg_input.txt', "r") as file:
        #with open(f'./LIFE/Sorg20X/Sorg20X_psg_input.txt', "r") as file:
            lines = file.readlines()

        rows = []
        for line in lines:
            if "<ATMOSPHERE-LAYER-" in line:
                index = line.index(">")
                rows.append(line[index+1:-1]) # remove \n from end of row

        columns = [ "Pressure", "Temperature", "Altitude", "H2", "He", "H2O", "CH4", "C2H6", "CO2", "C2H2", "C2H4", "CO",
                    "H2CO", "NH3", "SO2", "H2S", "SO", "CS2", "OCS", "DMS", "C2H6S2"]
        #
        if self.name!='Sorg20X': #and "_" not in self.name:
            columns=[ "Pressure", "Temperature", "Altitude", "H2", "He",'DMS']
        #else:
            #columns = [ "Pressure", "Temperature", "Altitude", "H2", "He", "DMS"]
            
        df = pd.DataFrame([row.split(",") for row in rows])
        df.columns = columns
        df = df.astype(float) # Convert all columns to float
        df = df.iloc[::-1] # reverse order, bc pRT reads temps from top to bottom of atmosphere

        return df

psg = PSG_input(obj).table
temperature = PSG_input(obj).temperature#[::2]
pressure = PSG_input(obj).pressure#[::2]
gravity = 1243 # cm / s²

LIFE_dict={'log_g':(np.log10(gravity),r'log $g$')}

def log_gradient_at_five_points(pressure, temperature):
    pressure = np.asarray(pressure)
    temperature = np.asarray(temperature)
    lnP = np.log(pressure)
    lnT = np.log(temperature)
    dlnT_dlnP_full = np.gradient(lnT, lnP)
    indices = np.linspace(0, len(pressure) - 1, 5, dtype=int)
    return dlnT_dlnP_full[indices]

dlnT_dlnP = log_gradient_at_five_points(pressure, temperature)#[::-1] #i think reverse order is correct?
for i in range(len(dlnT_dlnP)):
    LIFE_dict[f'dlnT_dlnP_{i}'] = (dlnT_dlnP[i],fr'$\nabla T_{i}$')
LIFE_dict['T0'] = (temperature[-1], r'$T_0$') # at bottom of atmosphere

VMR_maxcont = load_pickle('LIFE/Sorg20X/VMR_maxcont.pickle')
for species_i in species_names:
    LIFE_dict[f'log_{species_i}'] = (VMR_maxcont[f'log_{species_i}'],f'log {species_i}')

LIFE_parameters = {}
LIFE_mathtext = {}
for key_i, (value_i, mathtext_i) in LIFE_dict.items():
   LIFE_parameters[key_i] = value_i
   LIFE_mathtext[key_i] = mathtext_i

# only execute code if run directly from terminal, otherwise just import params dict
if __name__ == "__main__":

    from astropy import constants as const
    from astropy import units as u
    import pathlib
    import matplotlib.pyplot as plt
    import pickle
    from copy import deepcopy
    from scipy.ndimage import gaussian_filter
    from petitRADTRANS import Radtrans
    from io import StringIO

    n_atm_layers=100
    species_pRT=[]
    for species_i in species_names:
        if species_i not in ['H2','He']:
            species_pRT.append(species_info.loc[species_i,'pRT_name'])
        
    VMRs = {}
    for species_i in species_names:
        VMRs[species_i] = psg[species_i].values

    def pad_H2_VMR():
        VMR_wo_H2 = np.zeros(n_atm_layers)
        mass_fractions = {} # Create a dictionary for all used species
        for species_i in species_names:

            if species_i=='H2':
                continue

            species_pRT_i = species_info.loc[species_i,'pRT_name']
            #mass_i = species_info.loc[species_i,'mass']
            VMR_i = psg[species_i].values
            #mass_fractions[species_pRT_i] = mass_i * VMR_i
            VMR_wo_H2 += VMR_i
        VMR_H2 = np.ones(n_atm_layers)-VMR_wo_H2
        return VMR_H2

    VMRs['H2']=pad_H2_VMR()
    #print(VMRs['H2'].shape)
    # check spectrum w retrieval abunds
    #VMR_dict_retrieval = './LIFE/Sorg20X/varchem_PTgrad_N407_ev0.001/VMR_dict.pickle'
   # VMR_dict_retrieval = load_pickle(VMR_dict_retrieval)
   # for spec in species_names:
      #  if spec in VMR_dict_retrieval.keys():
      #      VMRs[spec] = np.median(np.array(VMR_dict_retrieval[spec]),axis=0)
     #   else:
   #         VMRs[spec]=VMRs[spec][::2]*1e-10 
    #print(np.median(np.array(VMRs['H2']),axis=0).shape)
   # n_atm_layers=50
   # gravity=10**3.09
   # pressure= pressure[::2]
   # temperature= temperature[::2]

    def VMR_to_MF(VMR_dict):
        MMW = 0.
        for species_i, VMR_i in VMR_dict.items():
            mass_i = species_info.loc[species_i,'mass']
            MMW += mass_i * VMR_i

        mass_fractions = {'MMW': MMW * np.ones(n_atm_layers)}
        for species_i, VMR_i in VMR_dict.items():            
            species_pRT_i = species_info.loc[species_i,'pRT_name']
            mass_i = species_info.loc[species_i,'mass']
            mass_fractions[species_pRT_i] = VMR_i * mass_i/MMW
        return mass_fractions, MMW

    atm_file=pathlib.Path('./LIFE/Sorg20X/atmosphere_objects_input.pickle')
    if atm_file.exists():
        atmosphere=load_pickle(atm_file)
    #if True:
    else:
        atmosphere = Radtrans(line_species=species_pRT,
                            rayleigh_species = ['H2','He'],
                            continuum_opacities = CIA_species,
                            wlen_bords_micron=np.array([np.min(wl_R50),np.max(wl_R50)]), 
                            mode='c-k')
        save_pickle(atmosphere,'./LIFE/Sorg20X/atmosphere_objects_input.pickle')

    wave_cm, opas = atmosphere.get_opa(np.array([300]).reshape(1))
    atmosphere.setup_opa_structure(pressure)
    mass_fractions, MMW = VMR_to_MF(VMRs)
    #print(MMW,mass_fractions['H2O_HITEMP'])
    #MMW = np.ones(n_atm_layers)*5.261996961574364
    #mass_fractions['MMW']=MMW
    #print(MMW) # any around 5.2?? no, maximum 4.2

    atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, contribution=True)
    wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-5 # cm
    flux = atmosphere.flux*const.c.to(u.km/u.s).value/(wl**2) # convert from flux density to f
    #print('max flux',np.nanmax(flux))

    def convolve_to_resolution(in_wlen, in_flux, out_res, in_res=None):
            
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

    #[erg cm^{-2} s^{-1} cm^{-1}]
    wl_model_R1000 = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
    # [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
    fl_model_R1000 = atmosphere.flux * const.c / (wl_model_R1000**2)

    fl_model_R50 = convolve_to_resolution(wl_model_R1000,fl_model_R1000,50)
    fl_model_R50/=np.max(fl_model_R50)
    fl_model_R50 = np.interp(wl_R50, wl_model_R1000, fl_model_R50) # interp into data wl
    
    ################################################################
    def read_file(file):
        file=np.genfromtxt(f'LIFE/{file}',skip_header=1,delimiter='	')
        wl=file[:,0]
        fl=file[:,1]
        snr=file[:,2]
        err=fl/snr
        return wl, fl, err
    n=20
    # original flux in [photons s^-1 m^-3]
    wl_LIFEsim, fl_LIFEsim, err_LIFEsim = read_file(f'Sorg{n}X/k18b_{n}x_LIFEsim_DA_05032025.txt')

    def pRT_to_photon_flux(atmosphere):
        nu = atmosphere.freq * u.Hz # Frequency grid from pRT
        wl_um = (const.c / nu).to(u.um)
        wl_m = wl_um.to(u.m)
        # pRT output is per Hz, convert to per micron:
        flux_nu = atmosphere.flux * u.erg / (u.cm**2 * u.s * u.Hz)

        # dν/dλ = -c / λ²  ⇒ |dν/dλ| = c / λ²
        dnu_dlambda = (const.c / wl_m**2).to(u.Hz / u.um)
        flux_lambda = (flux_nu * dnu_dlambda).to(u.erg / (u.cm**2 * u.s * u.um))

        E_photon = (const.h * const.c / wl_m).to(u.J)
        flux_J_m2_s_um = flux_lambda.to(u.J / u.m**2 / u.s / u.um) # Convert flux to J/m²/s/μm

        # Photon flux [photons / s / m² / μm]
        flux_photon = (flux_J_m2_s_um / E_photon).to(1 / (u.s * u.m**2 * u.um))

        R_earth = 6.371e6  # meters
        R_p = 2.61 * R_earth * u.m
        d_p = (5 * u.pc).to(u.m)

        scaling = (R_p / d_p) ** 2
        flux_observed = flux_photon*scaling*4*np.pi
        t_obs = 24 * 3600  # seconds
        flux_observed*= t_obs
        return wl_um.value,flux_observed

    wl_um,flux_observed = pRT_to_photon_flux(atmosphere)
    flux_observed = convolve_to_resolution(wl_um,flux_observed,50)
    flux_observed = np.interp(wl_LIFEsim, wl_um, flux_observed)
    
    plt.plot(wl_LIFEsim,fl_LIFEsim,c='k')
    plt.plot(wl_LIFEsim,flux_observed,c='r')
    plt.savefig('./LIFE/Sorg20X/flux_photon_units.png',dpi=200)
    ################################################################

    fl_model_R1000/=np.max(fl_model_R1000)
    fl_model_R1000 = np.interp(wl_R1000, wl_model_R1000, fl_model_R1000) # interp into data wl

    xmin, xmax= np.min(temperature), np.max(temperature)
    summed_contr=np.nansum(atmosphere.contr_em,axis=1)
    contribution_plot =summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin

    #psg_contr=np.genfromtxt(f'LIFE/Sorg20X/Sorg20X_contr.txt',skip_header=1,delimiter=' ')
    def get_contr(objec):
        psg_contr=np.genfromtxt(f'LIFE/Sorg20X/{objec}_contr.txt',skip_header=1,delimiter=' ')
        psg_contr=psg_contr[:,1:] # exclude first column (wavelength)
        psg_contr=np.sum(psg_contr,axis=0)[::-1] # sum over all wavelengths, change order
        psg_contr =psg_contr/np.max(psg_contr)*(xmax-xmin)+xmin
        return psg_contr

    psg_contr = get_contr(obj)

    idx_maxcont=np.where(summed_contr == np.max(summed_contr))[0][0]
    T_maxcont = temperature[idx_maxcont]
    P_maxcont = pressure[idx_maxcont]
    print(T_maxcont,np.log10(P_maxcont))

    # make spectrum with abundances at maxcont
    VMRs2 = VMRs.copy()
    VMR_wo_H2=0.
    for species_i in species_names:
        if species_i not in ['H2','He']:
            vmr = VMRs[species_i][idx_maxcont]
            VMR_wo_H2+=vmr
            VMRs2[species_i].fill(vmr) 
    VMRs2['He'].fill(0.15)
    VMRs2['H2'].fill(1-(VMR_wo_H2+0.15))

    mass_fractions2, MMW2 = VMR_to_MF(VMRs)
    atmosphere2 = deepcopy(atmosphere)
    atmosphere2.calc_flux(temperature, mass_fractions2, gravity, MMW2, contribution=True)
    wl_model_R1000_constVMR = const.c.to(u.km/u.s).value/atmosphere2.freq/1e-9 # mircons
    fl_model_R1000_constVMR = atmosphere2.flux * const.c / (wl_model_R1000_constVMR**2)

    fl_model_R50_constVMR = convolve_to_resolution(wl_model_R1000_constVMR,fl_model_R1000_constVMR,50)
    fl_model_R50_constVMR/=np.max(fl_model_R50_constVMR)
    fl_model_R50_constVMR = np.interp(wl_R50, wl_model_R1000_constVMR, fl_model_R50_constVMR) # interp into data wl

    fl_model_R1000_constVMR/=np.max(fl_model_R1000_constVMR)
    fl_model_R1000_constVMR = np.interp(wl_R1000, wl_model_R1000_constVMR, fl_model_R1000_constVMR) # interp into data wl

    # save VMR of each species at maxcont
    VMR_maxcont={}
    for species_i in species_names:
        VMR_maxcont[f'log_{species_i}'] = np.log10(psg[species_i].values[idx_maxcont])
        #VMR_maxcont[f'log_{species_i}'] = np.log10(np.max(psg[species_i].values))
    #save_pickle(VMR_maxcont,'LIFE/Sorg20X/VMR_maxcont.pickle')
    #save_pickle(VMR_maxcont,'LIFE/Sorg20X/VMR_maxtotal.pickle')

    # 3.420676034483222 3.029884415683024 3.139170834615479
    print(MMW[idx_maxcont],np.mean(MMW),np.median(MMW))
    wave_um = wave_cm*1e4

    fig,axes=plt.subplots(2,2,figsize=(7,4),dpi=200,gridspec_kw={'width_ratios':[2,0.7]})
    ax, ax3, ax2, ax4 = axes.flatten()

    ax4.axis('off')

    ax3.plot(temperature,pressure,alpha=1,c='k',label='Input')
    ax3.plot(psg_contr,pressure,alpha=0.5,c='k',linestyle='dashdot',label='PSG contr')

    ax3.plot(contribution_plot,pressure,alpha=0.5,c='darkturquoise',linestyle='dotted',label='pRT contr')
    ax3.set_xlim(xmin,xmax)
    ax3.legend(fontsize=7)
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Pressure [bar]')
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_yscale('log')
    ax3.invert_yaxis()

    maxval=[]
    for name in species_names:
        if name not in ['H2','He','C2H6S2']:
            prt = species_info.loc[name,'pRT_name']
            opa =opas[prt]*VMRs[name][idx_maxcont]
            maxval.append(opa)
            spec,=ax.plot(wave_um,opa,lw=0.5,
                        c=species_info.loc[name,'color'],label=species_info.loc[name,'mathtext_name'])
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ymin,ymax=np.max(maxval),1e-8
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Opacity [cm$^2$/g]')
    handles, labels = ax.get_legend_handles_labels()
    leg = ax4.legend(handles, labels, loc="center",ncol=2,fontsize=6)
    for line in leg.get_lines():
        line.set_linewidth(3)

    #fl_scaled = scale_between(ymin,ymax,fl_input)
    ax2.plot(wl_R1000,fl_R1000,c='orange',lw=0.8,alpha=0.8,label='PSG $\mathcal{R}$=1000')
    ax2.plot(wl_R1000,fl_model_R1000,lw=0.8,alpha=0.8,c='darkturquoise',label='pRT $\mathcal{R}$=1000')
    ax2.plot(wl_R50,fl_R50,c='red',lw=0.8,alpha=0.8,label='PSG LIFESim $\mathcal{R}$=50')
    ax2.plot(wl_R50,fl_model_R50,lw=0.8,alpha=0.8,c='blue',label='pRT $\mathcal{R}$=50')
    ax2.set_xlim(np.min(wl_R50),np.max(wl_R50))
    #ax2.plot(wl_input,fl_model2,lw=0.8,alpha=0.8,c='blue',label='pRT const')

    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right',fontsize=7)
    ax2.set_ylabel('Normalized flux')
    ax2.set_xlabel("Wavelength [$\mathrm{\mu}$m]")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    saveas = f'{obj}_pRT_PSG.png'
    print(f'saving as LIFE/Sorg20X/{saveas}')
    plt.savefig(f'LIFE/Sorg20X/{saveas}')
    plt.close()

    def save_spectrum(wl,fl,err,suffix=''):
        spectrum=np.full(shape=(len(wl),3),fill_value=np.nan)
        spectrum[:,0]=wl
        spectrum[:,1]=fl
        spectrum[:,2]=err
        np.savetxt(f'./LIFE/Sorg20X/Sorg20X_spectrum{suffix}.txt',spectrum,delimiter=' ',header='wavelength (nm) flux flux_error')

    if False:
        save_spectrum(wl_R50,fl_model_R50,err_R50,'_pRT50')
        save_spectrum(wl_R1000,fl_model_R1000,err_R1000,'_pRT1000')
        save_spectrum(wl_R50,fl_model_R50_constVMR,err_R50,'_pRT50_constVMRs')
        save_spectrum(wl_R1000,fl_model_R1000_constVMR,err_R1000,'_pRT1000_constVMRs')

    fig,ax=plt.subplots(1,1,figsize=(5,3),dpi=200)

    #ax.plot(wl_R1000,fl_R1000,c='orange',lw=0.8,alpha=0.8,label='PSG $\mathcal{R}$=1000')
    ax.plot(wl_R1000,fl_model_R1000,lw=0.8,alpha=0.8,c='darkturquoise',label='pRT $\mathcal{R}$=1000')
    ax.plot(wl_R1000,fl_model_R1000_constVMR,lw=0.8,alpha=0.8,c='violet',label='pRT $\mathcal{R}$=1000 constVMRs')
    #ax.plot(wl_R50,fl_R50,c='red',lw=0.8,alpha=0.8,label='PSG LIFESim $\mathcal{R}$=50')
    ax.plot(wl_R50,fl_model_R50,lw=0.8,alpha=0.8,c='blue',label='pRT $\mathcal{R}$=50')
    ax.plot(wl_R50,fl_model_R50_constVMR,lw=0.8,alpha=0.8,c='m',label='pRT $\mathcal{R}$=50 constVMRs')
    ax.legend(loc='upper right',fontsize=8)
    ax.set_xlim(np.min(wl_R50),np.max(wl_R50))
    plt.savefig(f'LIFE/Sorg20X/spectra_const_VMRs.png')

    if False: # show DMS opacity
        wlen_range=np.array([3,20]) # in microns for pRT
        atmosphere_object = Radtrans(line_species=['DMS'],
                            rayleigh_species = ['H2', 'He'],
                            continuum_opacities = ['H2-H2', 'H2-He'],
                            wlen_bords_micron=wlen_range, 
                            mode='c-k')

        T = np.array([298]).reshape(1)
        wave_cm, opas = atmosphere_object.get_opa(T)
        wave_um = wave_cm*1e4 # microns

        with open('./LIFE/DMS_298K_1bar_psg_opa.txt', "r") as f:
            lines = [line.strip() for line in f if not line.startswith("#")]
        data = pd.read_csv(StringIO("\n".join(lines)), sep='\s+')  # Adjust delimiter if needed
        wl = np.array(data.iloc[:, 0])
        opa = np.array(data.iloc[:, 1])/62.13404*6.022*1e23 # same units

        fig,ax=plt.subplots(1,1,figsize=(6,3),dpi=100)
        plt.plot(wave_um,opas['DMS'],label='pRT 298K')
        plt.plot(wl,opa,label='PSG 298K')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Wavelength [um]')
        plt.ylabel('Opacity [cm$^2$/g]')
        plt.savefig('./LIFE/Sorg20X/DMS_opa.jpg',dpi=200)
        