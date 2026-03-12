import numpy as np
import pandas as pd
import os
import copy
from utils import *
os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"

CIA_species =  ['H2-H2','H2-He','CH4-CH4','CH4-He','CO2-CH4','CO2-CO2','CO2-H2','CO2-He','H2-CH4','H2O-H2O']
species_names=["H2", "He", "H2O", "CH4", "C2H6", "CO2", "C2H2", "C2H4", "CO",
                    "H2CO", "NH3", "SO2", "H2S", "SO", "CS2", "OCS","DMS","C2H6S2"]
species_info = pd.read_csv('species_info_ck.csv',index_col=0)
gravity = 1243 # cm / s²

def log_gradient_from_input(pressure_in, temperature_in, pressure_ret, 
                            n_pt, log_P_knots=None, retrun_grad_input=False):
    # arrays
    pressure_in = np.asarray(pressure_in)
    temperature_in = np.asarray(temperature_in)
    pressure_ret = np.asarray(pressure_ret)

    # work in log space
    logP_in = np.log10(pressure_in)
    logT_in = np.log10(temperature_in)
    logP_ret = np.log10(pressure_ret)
    if retrun_grad_input:
        dlnT_dlnP_in = np.gradient(logT_in, logP_in)
        return dlnT_dlnP_in

    # restrict input profile to retrieved pressure range
    mask = (logP_in >= logP_ret.min()) & (logP_in <= logP_ret.max())
    logP_in_cut = logP_in[mask]
    logT_in_cut = logT_in[mask]

    # interpolate input T onto retrieved pressure grid
    logT_interp = np.interp(logP_ret, logP_in_cut, logT_in_cut)

    # compute gradients on retrieved grid
    dlnT_dlnP_full = np.gradient(logT_interp, logP_ret)

    if log_P_knots is None:
        # choose n_pt equally spaced indices in logP space
        indices = np.linspace(0, len(logP_ret) - 1, n_pt, dtype=int)
    else:
        log_P_knots = np.asarray(log_P_knots)
        indices = []
        for lp in log_P_knots:
            # find closest logP_ret to this knot
            idx = np.argmin(np.abs(logP_ret - lp))
            indices.append(idx)
        indices = np.array(indices, dtype=int)
    return dlnT_dlnP_full[indices]

def get_input_dict(obj, P_ret, n_pt):

    temperature = PSG_input(obj).temperature
    pressure = PSG_input(obj).pressure
    LIFE_dict = {'log_g': (np.log10(gravity), r'log $g$')}
    dlnT_dlnP = log_gradient_from_input(pressure, temperature, P_ret, n_pt)[::-1] #i think reverse order is correct?
    for i in range(len(dlnT_dlnP)):
        LIFE_dict[f'dlnT_dlnP_{i}'] = (dlnT_dlnP[i],fr'$\nabla T_{i}$')
    LIFE_dict['T0'] = (temperature[-1], r'$T_0$') # at bottom of atmosphere

    VMR_maxcont = load_pickle(f'LIFE/{obj}/VMR_maxcont.pickle')
    for species_i in species_names:
        LIFE_dict[f'log_{species_i}'] = (VMR_maxcont[f'log_{species_i}'],f'log {species_i}')

    LIFE_parameters = {}
    LIFE_mathtext = {}
    for key_i, (value_i, mathtext_i) in LIFE_dict.items():
        LIFE_parameters[key_i] = value_i
        LIFE_mathtext[key_i] = mathtext_i

    return LIFE_parameters

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

    nx=1
    const_H2O = False
    units_for_LIFESim = True # need to be W/sr/m2/um
    few_species=False
    save_for_retrieval=False
    resolution=int(1000)
    plot_contribution_per_species = True

    #if units_for_LIFESim==True and few_species==False and save_for_retrieval==False:

    obj = f'Sorg{nx}X'
    psg = PSG_input(obj).table
    temperature = PSG_input(obj).temperature
    pressure = PSG_input(obj).pressure
    
    n_pt = 7
    skewed_nodes = True
    pressure_ret = np.logspace(0,-6,50)
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(5,3),dpi=100,sharey=True)

    def get_temp_gradients(n_pt, ax, ax2):
        
        if skewed_nodes:
            log_P_knots = generate_skewed_p_nodes(n_nodes=n_pt, skew=0.75)[::-1]
        else:
            log_P_knots = np.linspace(np.log10(np.max(pressure_ret)),
                                np.log10(np.min(pressure_ret)),num=n_pt)

        dlnT_dlnP_knots = log_gradient_from_input(pressure, temperature, pressure_ret,
                                                  n_pt, log_P_knots=log_P_knots)
        interp_func = interp1d(log_P_knots,dlnT_dlnP_knots,kind='quadratic') # for the other 50 atm layers
        dlnT_dlnP = interp_func(np.log10(pressure_ret)) # T0 & P0 at beginning, start at bottom of atm
        T_base = max(temperature) # bottom
        ln_P = np.log(pressure_ret)
        temperature_ret = [T_base, ]
        lower_T_lim = 10
        upper_T_lim = 500
        # calc temperatures relative to base pressure, from bottom to top of atmosphere
        for i, ln_P_up_i in enumerate(ln_P[1:]): # start after base, T at base already defined
            ln_P_low_i = ln_P[i]
            ln_T_low_i = np.log(temperature_ret[-1])
            # compute temperatures based on gradient
            ln_T_up_i = ln_T_low_i + (ln_P_up_i - ln_P_low_i)*dlnT_dlnP[i+1]
            next_T = np.exp(ln_T_up_i)
            next_T = np.clip(next_T, lower_T_lim, upper_T_lim)
            temperature_ret.append(next_T)
        temperature_ret = np.array(temperature_ret)
        alpha=0.8
        ax.plot(temperature_ret,pressure_ret,label=f'$n={int(n_pt)}$',alpha=alpha)
        line, = ax2.plot(dlnT_dlnP, pressure_ret,alpha=alpha)
        ax2.scatter(dlnT_dlnP_knots, 10**log_P_knots, color=line.get_color(),s=6,alpha=alpha)
        return
    
    dlnT_dlnP_in = log_gradient_from_input(pressure, temperature,
                            pressure_ret, n_pt, retrun_grad_input=True)    
    ax.plot(temperature,pressure,label='Input',c='indigo')
    ax.invert_yaxis()
    ax.set_yscale('log')
    ax.set_ylabel('Pressure [bar]')
    ax.set_xlabel('Temperature [K]')
    ax2.plot(dlnT_dlnP_in,pressure,label='Input',c='indigo')
    ax2.set_xlabel(r'$\nabla T$')

    for n_pt in [5,6,7,8]:
        get_temp_gradients(n_pt, ax, ax2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if skewed_nodes:
        plt.savefig(f'./LIFE/{obj}/PT_skewed_reconstucted.png',dpi=200)
    else:
        plt.savefig(f'./LIFE/{obj}/PT_reconstucted.png',dpi=200)
    plt.close()

    suff = ''
    if const_H2O:
        suff = '_constH2O'
    if few_species:
        suff = '_few'

    # original flux in [photons s^-1 m^-3]
    def read_file(file):
        file=np.genfromtxt(f'LIFE/{file}',skip_header=1,delimiter='	')
        wl=file[:,0]
        fl=file[:,1]
        snr=file[:,2]
        return wl, fl, snr
    #wl_LIFEsim, fl_LIFEsim, err_LIFEsim = read_file(f'Sorg{nx}X/k18b_{nx}x_LIFEsim_DA_05032025.txt')
    
    # load in wavelength grid of desired resolution (if not 1000)
    res=50 if resolution not in [50,100] else resolution
    wl_LIFEsim, fl_LIFEsim, snr_LIFEsim = read_file(f'Sorg{nx}X/LIFEsim_R{res}_{nx}x.txt')
    err_LIFEsim = fl_LIFEsim/snr_LIFEsim

    n_atm_layers=100
    species_pRT=[]
    for species_i in species_names:
        if species_i not in ['H2','He']:
            species_pRT.append(species_info.loc[species_i,'pRT_name'])
        
    VMRs = {}
    for species_i in species_names:
        VMRs[species_i] = psg[species_i].values

    if const_H2O: # set H2O abund constant, at value in photosphere
        h2o_vmr = 0.023412
        print('Using constant VMR for H2O of',h2o_vmr)
        VMRs['H2O'] = np.ones_like(VMRs['H2O'])*h2o_vmr

    if few_species:
        print('Using few species')
        for species_i in species_names:
            if species_i not in ['H2O','CO2']:
                VMRs[species_i] = np.ones_like(VMRs[species_i])*1e-12

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
    if atm_file.exists() and False:
        atmosphere=load_pickle(atm_file)
    else:
        atmosphere = Radtrans(line_species=species_pRT,
                            rayleigh_species = ['H2','He'],
                            continuum_opacities = CIA_species,
                            wlen_bords_micron=np.array([np.min(wl_LIFEsim),np.max(wl_LIFEsim)]), 
                            mode='c-k')
        save_pickle(atmosphere,'./LIFE/Sorg20X/atmosphere_objects_input.pickle')

    wave_cm, opas = atmosphere.get_opa(np.array([300]).reshape(1))
    atmosphere.setup_opa_structure(pressure)
    mass_fractions, MMW = VMR_to_MF(VMRs)
    #MMW = np.ones(n_atm_layers)*5.261996961574364
    #mass_fractions['MMW']=MMW
    #print(MMW) # any around 5.2?? no, maximum 4.2

    atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, contribution=True)
    wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-5 # cm
    flux = atmosphere.flux*const.c.to(u.km/u.s).value/(wl**2) # convert from flux density to f

    def convolve_to_resolution(in_wlen, in_flux, out_res, in_res=None):

        if isinstance(in_wlen, u.Quantity):
            in_wlen = in_wlen.to(u.nm).value

        if in_res is None:
            in_res = np.mean(in_wlen[:-1] / np.diff(in_wlen))

        sigma_LSF = np.sqrt(1./out_res**2 - 1./in_res**2) / (2.*np.sqrt(2.*np.log(2.)))
        spacing = np.mean(2.*np.diff(in_wlen)/(in_wlen[1:] + in_wlen[:-1]))
        sigma_pix = sigma_LSF / spacing

        return gaussian_filter(in_flux, sigma=sigma_pix, mode='reflect')

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

    wl_model_R1000 = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
    # [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
    wl_cm = wl_model_R1000 * 1e-4  # µm → cm
    fl_model_R1000 = atmosphere.flux * const.c.cgs.value / (wl_cm**2)

    def pRT_to_PSG_units(flux): # for LIFESim
        # erg/cm2/s/cm -> W/sr/m2/um
        # 1 erg/s = 1e-7 W
        # 1 / cm^2 = 1 / (1e-2m)^2 = 1/ 1e-4 m^2 = 1e4 m^2
        # 1/cm = 1/ (1e-2 m) = 1/(1e-2 (1e6 um)) = 1/(1e4 um) = 1e-4 / um
        flux_new = flux*1e-7*1e4*1e-4/np.pi
        return flux_new
    
    if save_for_retrieval:
        wl_model_R1000,fl_model_R1000 = pRT_to_photon_flux(atmosphere)
    
    if resolution==1000:
        wavelength = wl_model_R1000
        fluxx = fl_model_R1000
    else:
        flux_conv = convolve_to_resolution(wl_model_R1000,fl_model_R1000,resolution)
        wavelength = wl_LIFEsim
        fluxx = np.interp(wavelength, wl_model_R1000, flux_conv) # interp into data wl
        
    ncol=2
    if save_for_retrieval:
        # use errors from existing model
        ncol=3
        if resolution==1000:
            print('getting R=1000 error')
            # get error similar to lifesim but for R1000
            err_R1000 = np.interp(wavelength, wl_LIFEsim, err_LIFEsim)
            err_LIFEsim = err_R1000
    spectrum=np.full(shape=(len(wavelength),ncol),fill_value=np.nan)
    spectrum[:,0]=wavelength

    if units_for_LIFESim:
        spectrum[:,1]=pRT_to_PSG_units(fluxx)
        savespec = f'./LIFE/Sorg{nx}X/Sorg{nx}X_pRT_spectrum_R{resolution}{suff}_newunits.txt'
        np.savetxt(savespec,spectrum,delimiter=' ',header='wavelength[um] flux[W/sr/m2/um]')
    else:
        spectrum[:,1]=fluxx
        savespec = f'./LIFE/Sorg{nx}X/Sorg{nx}X_pRT_spectrum_R{resolution}{suff}.txt'
        if save_for_retrieval:
            spectrum[:,2]=fluxx/err_LIFEsim
            savespec = f'./LIFE/Sorg{nx}X/LIFEsim_R1000_1x.txt'
        np.savetxt(savespec,spectrum,delimiter=' ',header='wavelength[um] flux[erg/cm2/s/cm]')
    print('Saving spectrum as:',savespec)

    fl_model_R50 = convolve_to_resolution(wl_model_R1000,fl_model_R1000,50)
    fl_model_R50 = np.interp(wl_LIFEsim, wl_model_R1000, fl_model_R50) # interp into data wl

    wl_um,flux_pRT_photons_R1000 = pRT_to_photon_flux(atmosphere)
    flux_pRT_photons_R50 = convolve_to_resolution(wl_um,flux_pRT_photons_R1000,50)
    flux_pRT_photons_R50 = np.interp(wl_LIFEsim, wl_um, flux_pRT_photons_R50)
    
    plt.plot(wl_LIFEsim,fl_LIFEsim,c='k')
    plt.plot(wl_LIFEsim,flux_pRT_photons_R50,c='r')
    plt.savefig(f'./LIFE/{obj}/flux_photon_units.png',dpi=200)
    ################################################################

    xmin, xmax= np.min(temperature), np.max(temperature)
    summed_contr=np.nansum(atmosphere.contr_em,axis=1)
    if units_for_LIFESim==True and few_species==False and save_for_retrieval==False:
        np.save(f'LIFE/{obj}/input_pRT_summed_contr.npy',summed_contr)
    contribution_plot =summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    
    idx_maxcont=np.where(summed_contr == np.max(summed_contr))[0][0]
    T_maxcont = temperature[idx_maxcont]
    P_maxcont = pressure[idx_maxcont]
    print('Max emission contribution: T=',T_maxcont,', log10(P)=',np.log10(P_maxcont))

    # make spectrum with abundances at maxcont
    VMRs2 = copy.deepcopy(VMRs)
    VMR_wo_H2=0.
    for species_i in species_names:
        if species_i!='H2':
            vmr = VMRs[species_i][idx_maxcont]
            VMR_wo_H2+=vmr
            VMRs2[species_i].fill(vmr) 
    VMRs2['H2'].fill(1-(VMR_wo_H2+0.15))

    mass_fractions2, MMW2 = VMR_to_MF(VMRs2)
    atmosphere2 = deepcopy(atmosphere)
    atmosphere2.calc_flux(temperature, mass_fractions2, gravity, MMW2, contribution=True)
    wl_model_R1000_constVMR = const.c.to(u.km/u.s).value/atmosphere2.freq/1e-9 # mircons
    fl_model_R1000_constVMR = atmosphere2.flux * const.c.cgs.value / (wl_cm**2)

    fl_model_R50_constVMR = convolve_to_resolution(wl_model_R1000_constVMR,fl_model_R1000_constVMR,50)
    fl_model_R50_constVMR = np.interp(wl_LIFEsim, wl_model_R1000_constVMR, fl_model_R50_constVMR) # interp into data wl

    # save VMR of each species at maxcont
    VMR_photosphere =pathlib.Path(f'LIFE/{obj}/VMR_maxcont.pickle')
    VMR_maxtotal =pathlib.Path(f'LIFE/{obj}/VMR_maxtotal.pickle')
    if VMR_photosphere.exists()==False and VMR_maxtotal.exists()==False:
        VMR_maxcont={}
        VMR_maxtot={}
        for species_i in species_names:
            VMR_maxcont[f'log_{species_i}'] = np.log10(psg[species_i].values[idx_maxcont])
            VMR_maxtot[f'log_{species_i}'] = np.log10(np.max(psg[species_i].values))
        if units_for_LIFESim==True and few_species==False and save_for_retrieval==False:
            save_pickle(VMR_maxcont,VMR_photosphere)
            save_pickle(VMR_maxtot,VMR_maxtotal)

    # 3.420676034483222 3.029884415683024 3.139170834615479
    #print(MMW[idx_maxcont],np.mean(MMW),np.median(MMW))
    wave_um = wave_cm*1e4

    fig,axes=plt.subplots(3,2,figsize=(7,6),dpi=200,gridspec_kw={'width_ratios':[2,0.8]})
    ax, ax6, ax2, ax4, ax5, ax3 = axes.flatten()

    ax6.axis('off')
    ax4.axis('off')
    if const_H2O:
        ax5.plot(psg['H2O'].values,
                 pressure,lw=1,alpha=0.5,linestyle='dashed',
                 c=species_info.loc['H2O','color'])
    for name in species_names:
        ax5.plot(VMRs[name],
                 pressure,lw=1,
                 c=species_info.loc[name,'color'])
    
    ax5.invert_yaxis()
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.set_xlim(1e-10,1e0)
    ax5.axhline(P_maxcont,lw=0.8,alpha=0.1,c='k')
    ax5.set_ylabel('Pressure [bar]')
    ax5.set_xlabel('VMR')

    ax3.plot(temperature,pressure,alpha=1,c='k',label='Input')
    ax3.axhline(P_maxcont,lw=0.8,alpha=0.1,c='k')
    ax3.axvline(T_maxcont,lw=0.8,alpha=0.1,c='k')
    ax3.plot(contribution_plot,pressure,alpha=0.5,c='k',linestyle='dotted',label='Phot.')
    ax3.set_xlim(xmin,xmax)
    ax3.legend(fontsize=8)
    ax3.set_xlabel('Temperature [K]')
    ax3.set_yticklabels([])
    ax3.yaxis.tick_right()
    ax3.set_yscale('log')
    ax3.invert_yaxis()

    maxval=[]
    for name in species_names:
        if name not in ['H2','He','C2H6S2']:
            prt = species_info.loc[name,'pRT_name']
            opa =opas[prt]*VMRs[name][idx_maxcont]
            maxval.append(opa)
            ax.plot(wave_um,opa,lw=0.5,
                        c=species_info.loc[name,'color'],label=species_info.loc[name,'mathtext_name'])
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ymin,ymax=np.max(maxval),1e-8
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Opacity [cm$^2$/g]')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel("Wavelength [$\mathrm{\mu}$m]")
    ax.xaxis.set_label_position('top') 
    handles, labels = ax.get_legend_handles_labels()
    leg = ax6.legend(handles, labels, loc="upper left",ncol=2,fontsize=8)
    for line in leg.get_lines():
        line.set_linewidth(3)

    ax2.plot(wl_um,flux_pRT_photons_R1000,lw=0.8,alpha=0.8,c='darkturquoise',label='pRT $\mathcal{R}$=1000')
    ax2.plot(wl_LIFEsim,fl_LIFEsim,c='red',lw=0.8,alpha=0.8,label='PSG LIFESim $\mathcal{R}$=50')
    ax2.plot(wl_LIFEsim,flux_pRT_photons_R50,lw=0.8,alpha=0.8,c='blue',label='pRT $\mathcal{R}$=50')
    ax2.set_xlim(np.min(wl_LIFEsim),np.max(wl_LIFEsim))

    ax2.grid(alpha=0.3)
    #ax2.legend(loc='upper right',fontsize=7)
    handles2, labels2 = ax2.get_legend_handles_labels()
    leg2 = ax4.legend(handles2, labels2, loc="upper left",fontsize=8)
    

    ax2.set_ylabel('Flux [photons s$^{-1}$ m$^{-3}$]')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    saveas = f'{obj}_pRT_PSG{suff}.png'
    print(f'Saving as LIFE/{obj}/{saveas}')
    plt.savefig(f'LIFE/{obj}/{saveas}')
    plt.close()

    if plot_contribution_per_species:
        from matplotlib.lines import Line2D

        def MF_MMW(nx):
            psg = PSG_input(f'Sorg{int(nx)}X').table
            VMRs = {}
            for species_i in species_names:
                VMRs[species_i] = psg[species_i].values
            mf, mmw = VMR_to_MF(VMRs)
            return mf, mmw
        mf1x, mmw1x =  MF_MMW(1)
        mf20x, mmw20x =  MF_MMW(20)
        bb_alpha = 0.5
        sorg_alpha=1

        def contribution_by_species_single(radtrans,
                                   temperatures, mass_fractions,
                                   gravity, MMW):

            # Full spectrum (for reference)
            radtrans.calc_flux(temperatures, mass_fractions, gravity, MMW,contribution=True)
            contr_em = radtrans.contr_em
            wl = const.c.to(u.km/u.s).value / radtrans.freq / 1e-5  # cm
            base_spec = radtrans.flux * const.c.cgs.value / (wl**2)
            #base_spec = pRT_to_PSG_units(base_spec)
            #_, base_spec = pRT_to_photon_flux(radtrans)
            base_spec = pRT_to_PSG_units(base_spec)

            contributions = {}

            for sp in mass_fractions.keys():
                # build mass fractions with ONLY this species
                mf = {}
                for k in mass_fractions:
                    if k == sp:
                        mf[k] = mass_fractions[k]
                    else:
                        mf[k] = 0.0 * mass_fractions[k]

                radtrans.calc_flux(temperatures, mf, gravity, MMW)
                spec_sp = radtrans.flux * const.c.cgs.value / (wl**2)
                #_, spec_sp = pRT_to_photon_flux(radtrans)
                spec_sp = pRT_to_PSG_units(spec_sp)
                contributions[sp] = spec_sp

            for sp in mass_fractions.keys():
                mf[sp] = 0.0 * mass_fractions[sp]
            radtrans.calc_flux(temperatures, mf, gravity, MMW)
            blackbody_flux = radtrans.flux * const.c.cgs.value / (wl**2)
            blackbody_flux = pRT_to_PSG_units(blackbody_flux)

            return base_spec, contributions, blackbody_flux, contr_em

        def make_atmosphere():
            atm = Radtrans(
                line_species=species_pRT,
                rayleigh_species=['H2','He'],
                continuum_opacities=CIA_species,
                wlen_bords_micron=np.array([np.min(wl_LIFEsim), np.max(wl_LIFEsim)]),
                mode='c-k'
            )
            atm.setup_opa_structure(pressure)
            return atm
        
        atm1 = make_atmosphere()
        tot1x, contribs1x, bb1x, contr_em1x = contribution_by_species_single(atm1, temperature, mf1x, gravity, mmw1x)
        atm2 = make_atmosphere()
        tot20x, contribs20x, bb20x, contr_em20x = contribution_by_species_single(atm2, temperature, mf20x, gravity, mmw20x)

        fig,axes=plt.subplots(4,1,figsize=(5,6),dpi=200)
        #flx = pRT_to_PSG_units(fluxx)
        #ax.plot(wl_um, flx, c='k',lw=0.8)
        ax,axc,ax2,ax2c = axes
        for a in [ax,ax2]:
            a.set_ylabel('Flux [W/sr/m$^2$/μm]')
  
        ax2.set_xlabel('Wavelength (μm)')
        wlmin, wlmax = min(wl_um),max(wl_um)
        for a in axes:
            a.set_xlim(wlmin, wlmax)
        X, Y = np.meshgrid(wl_um, pressure)

        #gamma = 1  # <1 enhances weak features, >1 suppresses them
        for a,contr in zip([axc,ax2c],[contr_em1x,contr_em20x]):
            #contr_scaled = contr**gamma  # element-wise power
            #a.imshow(contr_scaled, origin='lower', aspect='auto',
            #                    extent=[wlmin, wlmax, pressure.min(), pressure.max()],
            #                    cmap=plt.cm.bone_r, rasterized=True)
            a.contourf(X,Y,contr,30,cmap=plt.cm.bone_r)
            a.set_yscale('log')
            a.set_ylabel('Pressure [bar]')
            a.set_ylim(10**(-5.9),1)
            a.invert_yaxis()

        plot_species=['H2O','CH4','C2H6','CO2','C2H2','C2H4','CO','SO2','NH3',
                        'H2S','CS2','OCS','DMS','H2CO','SO','C2H6S2']
        plot_species=['H2O','CH4','C2H6','CO2','C2H4','CO','CS2','OCS','DMS']
        
        leg_y = 1.4
        #plot_species = ['H2O','CH4','C2H6','CO2','C2H4','CO','CS2','OCS','DMS']
        ncol=5
        def plot_contribs(tot,contribs,bb,axi):
            lines = []
            for sp, dp in contribs.items():

                # find matching label in species_names via pRT_name column
                match = species_info[species_info['pRT_name'] == sp]
                if match.empty:
                    continue  # sp not in table

                label = match.index[0]        # this is in species_names
                if label not in plot_species:
                    continue
                c = match.loc[label, 'color'] # or match['color'].iloc[0]
                mathtext = species_info.loc[label,'mathtext_name']
                lines.append(Line2D([0],[0],color=c,
                            linewidth=2,label=mathtext))

                axi.plot(wl_um, dp, label=label, c=c,lw=0.8)

            axi.plot(wl_um, tot, c='k',alpha=sorg_alpha,lw=0.8)
            axi.plot(wl_um, bb, c='k',linestyle='dotted',lw=3,alpha=bb_alpha)

            return lines
        
        lines = plot_contribs(tot1x,contribs1x,bb1x,ax)
        lines = plot_contribs(tot20x,contribs20x,bb20x,ax2)
        handles1 = [Line2D([0], [0], color='k',linewidth=2,alpha=sorg_alpha,label=r'1$\times\,\,$S$_\mathrm{org}$'),
                    Line2D([0], [0], color='k',linewidth=3,linestyle='dotted',alpha=bb_alpha,label='Continuum')]
        leg1 = ax.legend(handles=handles1, loc='upper right', frameon=False,handlelength=1.7)

        handles2 = [Line2D([0], [0], color='k',linewidth=2,alpha=sorg_alpha,label=r'20$\times\,\,$S$_\mathrm{org}$'),
                    Line2D([0], [0], color='k',linewidth=3,linestyle='dotted',alpha=bb_alpha,label='Continuum')]
        leg2 = ax2.legend(handles=handles2, loc='upper right', frameon=False,handlelength=1.7)

        species_leg = ax.legend(handles=lines,bbox_to_anchor=(0.5, leg_y),ncol=ncol,
                                loc='upper center',frameon=False,fontsize=9,handlelength=1.5)
        ax.add_artist(leg1)

        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        saveas = 'Opacity_contributions.pdf'
        print(f'Saving as LIFE/{obj}/{saveas}')
        plt.savefig(f'LIFE/{obj}/{saveas}')
        plt.savefig(f'LIFE/{obj}/Opacity_contributions.png')
        plt.close()

    if False:
        from matplotlib.lines import Line2D

        def MF_MMW(nx):
            psg = PSG_input(f'Sorg{int(nx)}X').table
            VMRs = {}
            for species_i in species_names:
                VMRs[species_i] = psg[species_i].values
            mf, mmw = VMR_to_MF(VMRs)
            return mf, mmw
        mf1x, mmw1x =  MF_MMW(1)
        mf20x, mmw20x =  MF_MMW(20)
        bb_alpha = 0.5
        sorg_alpha=1

        def contribution_by_species_single(radtrans,
                                   temperatures, mass_fractions,
                                   gravity, MMW):

            # Full spectrum (for reference)
            radtrans.calc_flux(temperatures, mass_fractions, gravity, MMW)
            wl = const.c.to(u.km/u.s).value / radtrans.freq / 1e-5  # cm
            base_spec = radtrans.flux * const.c.cgs.value / (wl**2)
            #base_spec = pRT_to_PSG_units(base_spec)
            #_, base_spec = pRT_to_photon_flux(radtrans)
            base_spec = pRT_to_PSG_units(base_spec)

            contributions = {}

            for sp in mass_fractions.keys():
                # build mass fractions with ONLY this species
                mf = {}
                for k in mass_fractions:
                    if k == sp:
                        mf[k] = mass_fractions[k]
                    else:
                        mf[k] = 0.0 * mass_fractions[k]

                radtrans.calc_flux(temperatures, mf, gravity, MMW)
                spec_sp = radtrans.flux * const.c.cgs.value / (wl**2)
                #_, spec_sp = pRT_to_photon_flux(radtrans)
                spec_sp = pRT_to_PSG_units(spec_sp)
                contributions[sp] = spec_sp

            for sp in mass_fractions.keys():
                mf[sp] = 0.0 * mass_fractions[sp]
            radtrans.calc_flux(temperatures, mf, gravity, MMW)
            blackbody_flux = radtrans.flux * const.c.cgs.value / (wl**2)
            blackbody_flux = pRT_to_PSG_units(blackbody_flux)

            return base_spec, contributions, blackbody_flux
        
        tot1x, contribs1x, bb1x = contribution_by_species_single(atmosphere, temperature, mf1x, gravity, mmw1x)
        tot20x, contribs20x, bb20x = contribution_by_species_single(atmosphere, temperature, mf20x, gravity, mmw20x)

        fig,(ax,ax2)=plt.subplots(2,1,figsize=(5,5),dpi=200,sharex=True)
        #flx = pRT_to_PSG_units(fluxx)
        #ax.plot(wl_um, flx, c='k',lw=0.8)
        ax.set_ylabel('Flux [W/sr/m$^2$/μm]')
        ax2.set_ylabel('Flux [W/sr/m$^2$/μm]')
        ax2.set_xlabel('Wavelength (μm)')
        ax.set_xlim(min(wl_um),max(wl_um))

        plot_species=['H2O','CH4','C2H6','CO2','C2H2','C2H4','CO','SO2','NH3',
                        'H2S','CS2','OCS','DMS','H2CO','SO','C2H6S2']
        plot_species=['H2O','CH4','C2H6','CO2','C2H4','CO','CS2','OCS','DMS']
        
        leg_y = 1.4
        #plot_species = ['H2O','CH4','C2H6','CO2','C2H4','CO','CS2','OCS','DMS']
        leg_y = 1.3
        ncol=5
        def plot_contribs(tot,contribs,bb,axi):
            lines = []
            for sp, dp in contribs.items():

                # find matching label in species_names via pRT_name column
                match = species_info[species_info['pRT_name'] == sp]
                if match.empty:
                    continue  # sp not in table

                label = match.index[0]        # this is in species_names
                if label not in plot_species:
                    continue
                c = match.loc[label, 'color'] # or match['color'].iloc[0]
                mathtext = species_info.loc[label,'mathtext_name']
                lines.append(Line2D([0],[0],color=c,
                            linewidth=2,label=mathtext))

                axi.plot(wl_um, dp, label=label, c=c,lw=0.8)

            axi.plot(wl_um, tot, c='k',alpha=sorg_alpha,lw=0.8)
            axi.plot(wl_um, bb, c='k',linestyle='dotted',lw=3,alpha=bb_alpha)

            return lines
        
        lines = plot_contribs(tot1x,contribs1x,bb1x,ax)
        lines = plot_contribs(tot20x,contribs20x,bb20x,ax2)
        #ax2.set_yscale('log')
        #ax.set_ylabel('Flux [photons s$^{-1}$ m$^{-3}$]',fontsize=11)
        #ax2.set_ylabel('Flux [photons s$^{-1}$ m$^{-3}$]',fontsize=11)
        #ax2.set_ylim(1e-3,600)
        #ax2.set_ylim(1,1e-1)
        #ax.text(0.03, 0.95, r'1$\times\,\,$S$_\mathrm{org}$',transform=ax.transAxes, ha="left", va="top")
        #ax2.text(0.03, 0.95, r'20$\times\,\,$S$_\mathrm{org}$',transform=ax2.transAxes, ha="left", va="top")
        handles1 = [Line2D([0], [0], color='k',linewidth=2,alpha=sorg_alpha,label=r'1$\times\,\,$S$_\mathrm{org}$'),
                    Line2D([0], [0], color='k',linewidth=3,linestyle='dotted',alpha=bb_alpha,label='Blackbody')]
        leg1 = ax.legend(handles=handles1, loc='upper right', frameon=False,handlelength=1.7)
        #ax.add_artist(leg1)

        handles2 = [Line2D([0], [0], color='k',linewidth=2,alpha=sorg_alpha,label=r'20$\times\,\,$S$_\mathrm{org}$'),
                    Line2D([0], [0], color='k',linewidth=3,linestyle='dotted',alpha=bb_alpha,label='Blackbody')]
        leg2 = ax2.legend(handles=handles2, loc='upper right', frameon=False,handlelength=1.7)
        #ax2.add_artist(leg2) 

        species_leg = ax.legend(handles=lines,bbox_to_anchor=(0.5, leg_y),ncol=ncol,
                                loc='upper center',frameon=False,fontsize=9,handlelength=1.5)
        ax.add_artist(leg1)

        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        saveas = 'Opacity_contributions.pdf'
        print(f'Saving as LIFE/{obj}/{saveas}')
        plt.savefig(f'LIFE/{obj}/{saveas}')
        plt.close()

    if False:
        # get transmission contribution for METIS project (not available wit pRT3)
        from petitRADTRANS import nat_cst as nc
        R_pl = 0.22* nc.r_jup_mean

        atmosphere = Radtrans(line_species=species_pRT,
                                    rayleigh_species = ['H2','He'],
                                    continuum_opacities = CIA_species,
                                    wlen_bords_micron=np.array([3.05,3.35]), 
                                    mode='c-k')
        atmosphere.setup_opa_structure(pressure)

        def get_contr(mf,mmw):
            atmosphere.calc_transm(temperature, mf,
                        gravity, mmw, R_pl=R_pl, P0_bar=0.1,
                        contribution = True)
            return atmosphere.contr_tr

        contr1x = get_contr(mf1x,mmw1x)
        np.save(f'LIFE/{obj}/transm_contr_1X.npy', contr1x)
        contr20x = get_contr(mf20x,mmw20x)
        np.save(f'LIFE/{obj}/transm_contr_20X.npy',contr20x)