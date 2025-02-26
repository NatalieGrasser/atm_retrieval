import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pathlib
import astropy.constants as const
from scipy.interpolate import interp1d
import warnings
warnings.simplefilter("error", RuntimeWarning)  # Convert warnings to exceptions

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

def rem_cont(wave,spec,poly_order=3):
    spec=spec.flatten()
    wave=wave.flatten()
    flux_contrem = np.full(spec.shape,np.nan)
    continuum_model = np.poly1d(np.polyfit(wave,spec,poly_order))
    continuum = continuum_model(wave)
    flux_contrem = spec/continuum      
    return flux_contrem

def get_ratios(retr_obj,equ_too=False): # in case not all ratios are in retrieval
    if retr_obj.chemistry in ['equchem','quequchem']:
        ratios_default = ['C/O','Fe/H']
        check_ratios =  ['log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_O16_18_ratio']
        for r in check_ratios:
            if r in retr_obj.parameters.param_keys:
                ratios_default.append(r)
    elif retr_obj.chemistry=='freechem':
        ratios_default = ['C/O','C/H']
        ratios_default_equ = ['C/O','Fe/H']
        check_ratios = ['log_12CO/13CO','log_12CO/C17O','log_12CO/C18O','log_H2O/H2(18)O']
        ratios_equ = ['log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_O16_18_ratio']
        check_molec = ['13CO','C17O','C18O','H2(18)O']
        for m,r,e in zip(check_molec,check_ratios,ratios_equ):
            if f'log_{m}' in retr_obj.parameters.param_keys:
                ratios_default.append(r)
                ratios_default_equ.append(e)
    if equ_too==False:
        return ratios_default
    else:
        return ratios_default, ratios_default_equ

# cross-correlate residuals with spectrum that contains selected species at equilibrium
def CCF_residuals(retr_obj,ccf_species,noiserange=100): # can only be run after evaluate()

    from retrieval import Retrieval
    from parameters import Parameters
    from pRT_model import pRT_spectrum
    import figures as figs

    ccf_dict={}
    ccf_acf_dict={} # save cross-correlations and auto-correlations

    if isinstance(ccf_species, list)==False:
        ccf_species=[ccf_species] # if only one, make list so that it works in the for loop

    RVs=np.arange(-500,500,1) # km/s
    for ccf_species_i in ccf_species:

        # create template with only selected species at equibilrium abundance
        parameters_spec = retr_obj.params_dict
        parameters_spec.update({'C/O': retr_obj.params_dict['C/O'],
                        'Fe/H': retr_obj.params_dict['C/H']})
        ratios_free,ratios_equ = get_ratios(retr_obj,equ_too=True)
        for r,e in zip(ratios_free,ratios_equ):
            parameters_spec.update({e: retr_obj.params_dict[r]})
        parameters_spec = Parameters({}, parameters_spec)
        parameters_spec.param_priors['log_l']=[-3,0]
        retr_spec = Retrieval(target=retr_obj.target,parameters=parameters_spec, 
                                species_names=retr_obj.species_names,Nlive=retr_obj.Nlive,
                                evtol=retr_obj.evtol,chemistry='equchem',
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

                if np.isnan(retr_obj.data_flux[order,det]).all():
                    pass # skip empty order/det, CCF and ACF remains 0 

                else:
                    
                    template_flux=template_fluxes[order]
                    template_wl = template_waves[order]
                    template_flux = rem_cont(template_wl,template_flux)
                    
                    wl_data=retr_obj.data_wave[order,det,retr_obj.mask_isfinite[order,det]]
                    fl_data = retr_obj.data_flux[order,det,retr_obj.mask_isfinite[order,det]]-retr_obj.model_flux[order,det,retr_obj.mask_isfinite[order,det]]
                    #plt.plot(template_wl,template_flux,c='tab:blue')
                    #plt.plot(wl_data,fl_data,c='tab:orange')
                    fl_data-=np.nanmean(fl_data)

                    retr_obj.Cov[order,det].get_cholesky() # in case it hasn't been called yet
                    cov_0_data=retr_obj.Cov[order,det].solve(fl_data)                            
                    wl_shift=wl_data[:, np.newaxis]*beta[np.newaxis, :]
                    template_shift=interp1d(template_wl,template_flux)(wl_shift) # interpolate template onto shifted wl
                    #template_shift-= np.nanmedian(template_shift)  
                    template_shift = np.array([template_shift[:,i] - np.nanmedian(template_shift[:,i]) for i in range(template_shift.shape[1])]).T
                    #print(order,det, np.nanmedian(template_shift),np.nanmedian(fl_data))
                    #temptemplate_shift[:,len(RVs)//2]

                    template_rebinned=interp1d(template_wl,template_flux)(wl_data)
                    template_rebinned-=np.nanmedian(template_rebinned)
                    cov_0_temp=retr_obj.Cov[order,det].solve(template_rebinned)

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
    figs.CCF_plot_all(retr_obj,ccf_species,noiserange=100,show_ACF=True,suffix='_res')

    return ccf_dict