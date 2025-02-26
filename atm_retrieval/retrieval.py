import getpass
import os
from pRT_model import pRT_spectrum
import figures as figs
from covariance import *
from log_likelihood import *
from target import Target
from utils import *

import numpy as np
import pymultinest
import pathlib
import pickle
from petitRADTRANS import Radtrans
import pandas as pd
import astropy.constants as const
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # pRT warning
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn') # occasional
warnings.simplefilter("error", RuntimeWarning)  # Convert warnings to exceptions
#warnings.filterwarnings("ignore", category=np.linalg.LinAlgError) 

class Retrieval:

    def __init__(self,target,parameters,species_names,Nlive,evtol,chemistry='freechem',
                 GP=True,cloud_mode='gray',PT_type='PTgrad',redo=False):
        
        self.Nlive=Nlive
        self.evtol=evtol
        for attr in ['primary_label','color1','color2','K2166']:
            setattr(self, attr, getattr(target, attr))
            
        self.target = target
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.spectral_resolution = target.calc_resolution()
        self.mask_isfinite=target.get_mask_isfinite() # mask nans, shape (orders,detectors)    
        self.separation,self.err_eff=target.prepare_for_covariance()
        self.parameters=parameters
        self.chemistry=chemistry # freechem/equchem/quequchem
        self.species_names=species_names
        self.species_pRT, self.species_hill =self.get_pRT_hill(species_names)

        # if companion, load in primary spectrum as well 
        if self.target.primary_label==False:
            self.target_primary=Target(f'{self.target.name[:-1]}A')
            self.primary_wave,self.primary_flux,self.primary_err=self.target_primary.load_spectrum()

        self.n_orders, self.n_dets, _ = self.data_flux.shape # shape (orders,detectors,pixels)
        self.n_params = len(parameters.free_params)
        self.output_name=f'{chemistry}_{PT_type}_N{Nlive}_ev{evtol}' # output folder name
        self.cwd = os.getcwd()
        self.output_dir = pathlib.Path(f'{self.cwd}/{self.target.name}/{self.output_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # cloud properties
        self.cloud_mode=cloud_mode
        self.do_scat_emis=False # only relevant for physical clouds (e.g. MgSiO3)
        self.cloud_species=None
        if cloud_mode=='MgSiO3':
            self.cloud_species=['MgSiO3(c)_cd']
            self.do_scat_emis = True # enable scattering on cloud particles
        self.PT_type=PT_type
        self.lbl_opacity_sampling=3
        self.n_atm_layers=50
        self.pressure = np.logspace(-6,2,self.n_atm_layers)  # like in deRegt+2024

        self.Cov = np.empty((self.n_orders,self.n_dets), dtype=object) # covariance matrix
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                mask_ij = self.mask_isfinite[i,j] # only finite pixels
                if not mask_ij.any(): # skip empty order/detector pairs
                    continue
                if GP==True: # use Gaussian processes covariance matrix
                    maxval=10**(self.parameters.param_priors['log_l'][1])*3 # 3*max value of prior of l
                    self.Cov[i,j] = CovGauss(err=self.data_err[i,j,mask_ij],separation=self.separation[i,j], 
                                            err_eff=self.err_eff[i,j],max_separation=maxval)
                if GP==False: # use simple diagonal covariance matrix
                    self.Cov[i,j] = Covariance(err=self.data_err[i,j,mask_ij])
    
        self.LogLike = LogLikelihood(retr_obj=self,scale_flux=True,scale_err=True)

        # redo atmosphere objects when introdocuing new species or MgSiO3 clouds
        self.atmosphere_objects=self.get_atmosphere_objects(redo=redo)
        self.callback_label='live_' # label for plots
        self.prefix='pmn_'

        # will be updated, but needed as None until then
        self.bestfit_params=None 
        self.posterior = None
        self.params_dict=None

    def get_pRT_hill(self,species_names): # get pRT species name and hill notations
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        species_pRT=[] # pRT names
        species_hill=[] # hill notation
        for species_i in species_names:
            species_pRT.append(species_info.loc[species_i,'pRT_name'])
            species_hill.append(species_info.loc[species_i,'Hill_notation'])
        return species_pRT, species_hill

    def get_atmosphere_objects(self,redo=False,broader=True,for_species=None):

        atmosphere_objects=[]
        species=self.species_pRT if for_species==None else for_species
        if for_species!=None:
            species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
            species = [species_info.loc[species,'pRT_name']]
        if for_species==None: # none specified
            file=pathlib.Path('atmosphere_objects.pickle')
            not_exists=False
            if self.target.name=='ROXs12A': # different file for hotter objects, has additional species
                file=pathlib.Path('ROXs12A/atmosphere_objects.pickle')
            if self.target.name=='ROXs12B':
                file=pathlib.Path('ROXs12B/atmosphere_objects.pickle')
            if file.exists() and redo==False:
                atmosphere_objects= load_pickle(file)
                return atmosphere_objects
            else:
                not_exists=True
        if for_species!=None or not_exists:
            for order in range(self.n_orders):
                wl_pad=7 # wavelength padding because spectrum is not wavelength shifted yet
                if broader==True:  # larger wl pad needed when shifting during cross-correlation
                    rv_max = 501 # maximum RV for cross-corr
                    wl_max= np.max(self.K2166)
                    wl_pad = 1.1*rv_max/(const.c.to('km/s').value)*wl_max
                wlmin=np.min(self.K2166[order])-wl_pad
                wlmax=np.max(self.K2166[order])+wl_pad
                wlen_range=np.array([wlmin,wlmax])*1e-3 # nm to microns

                atmosphere = Radtrans(line_species=species,
                                    rayleigh_species = ['H2', 'He'],
                                    continuum_opacities = ['H2-H2', 'H2-He'],
                                    wlen_bords_micron=wlen_range, 
                                    mode='lbl',
                                    cloud_species=self.cloud_species,
                                    do_scat_emis=self.do_scat_emis,
                                    lbl_opacity_sampling=self.lbl_opacity_sampling) # take every nth point (=3 in deRegt+2024)
                
                atmosphere.setup_opa_structure(self.pressure)
                atmosphere_objects.append(atmosphere)
            if for_species==None:
                save_pickle(atmosphere_objects,file)
            return atmosphere_objects

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(self)      
        self.model_flux=self.model_object.make_spectrum()
        for j in range(self.n_orders): # update covariance matrix
            for k in range(self.n_dets):
                if not self.mask_isfinite[j,k].any(): # skip empty order/detector
                    continue
                self.Cov[j,k](self.parameters.params)
        ln_L = self.LogLike(self.model_flux, self.Cov, params=self.parameters.params) # retrieve log-likelihood
        return ln_L

    def PMN_run(self,N_live_points=400,evidence_tolerance=0.5,resume=True):
        pymultinest.run(LogLikelihood=self.PMN_lnL,Prior=self.parameters,n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'{self.output_dir}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=True, sampling_efficiency = 0.5,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # default is 0.5, high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=100)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        posterior = posterior[:,:-2] # remove last 2 columns
        posterior_dict={}
        for key in self.parameters.free_params.keys():
            idx=list(self.parameters.params).index(key)
            posterior_dict[key]=(posterior[:,idx],self.parameters.free_params[key][1])
        self.posterior=posterior_dict
        self.params_dict,self.model_flux=self.get_params_and_spectrum()
        figs.summary_plot(self)
        if self.chemistry in ['equchem','quequchem']:
            figs.VMR_plot(self)
        if self.primary_label==False: 
            figs.plot_spectrum_split(self,plot_components=True)
        else:
            figs.plot_spectrum_split(self)
     
    def PMN_analyse(self):

        # save posterior as dictionary with key: (posterior, mathtext)
        prefix = self.callback_label if self.callback_label!='final_' else ''
        post=pathlib.Path(f'{self.output_dir}/{prefix}posterior_dict.pickle')
        if post.exists():
            self.posterior=load_pickle(post)
        else:
            analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                            outputfiles_basename=f'{self.output_dir}/{self.prefix}')  # set up analyzer object
            stats = analyzer.get_stats()
            posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
            posterior = posterior[:,:-1] # shape 

            posterior_dict={}
            for key in self.parameters.free_params.keys():
                idx=list(self.parameters.params).index(key)
                posterior_dict[key]=(posterior[:,idx],self.parameters.free_params[key][1])
            self.posterior=posterior_dict
            save_pickle(self.posterior,post)
            self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # read params of best-fitting model, highest likelihood
            if self.prefix=='pmn_':
                self.lnZ = stats['nested importance sampling global log-evidence']
                print(f"\nFinal lnZ = {self.lnZ}\n")
            else: # when doing exclusion retrievals
                self.lnZ_ex = stats['nested importance sampling global log-evidence']

    def get_quantiles(self,posterior): # input only one posterior
        quantiles = np.array([np.percentile(posterior, [16.0,50.0,84.0])])
        median=quantiles[:,1][0] # median
        plus_err=quantiles[:,2][0]-median # +error
        minus_err=quantiles[:,0][0]-median # -error
        return median,minus_err,plus_err

    def get_params_and_spectrum(self): 

        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists():
            self.params_dict=load_pickle(final_dict)

            for key in self.parameters.param_keys:
                self.parameters.params[key]=self.params_dict[key] # set parameters to retrieved values

            # create final spectrum
            self.model_object=pRT_spectrum(self,contribution=True)
            self.model_flux0=self.model_object.make_spectrum()
            self.model_flux=np.zeros_like(self.model_flux0)
            self.summed_contr=np.nanmean(self.model_object.contr_em_orders,axis=0) # average over all orders
            phi_ij=self.params_dict['phi_ij']
            for order in range(self.n_orders):
                for det in range(self.n_dets):
                    self.model_flux[order,det]=phi_ij[order,det]*self.model_flux0[order,det] # scale model accordingly
            self.get_ratios() 

        else:
            # create dict of constant params + evaluated params + their errors
            self.params_dict=self.parameters.constant_params.copy() # initialize dict with constant params
            for key in self.parameters.param_keys:
                median,minus_err,plus_err = self.get_quantiles(self.posterior[key][0])
                self.params_dict[key]= median # add median of evaluated params (more robust than bestfit)
                self.params_dict[f'{key}_err']=(minus_err,plus_err)
                self.parameters.params[key]=median # set parameters to retrieved values

            # create final spectrum
            self.model_object=pRT_spectrum(self,contribution=True)
            self.model_flux0=self.model_object.make_spectrum()
            self.summed_contr=np.nanmean(self.model_object.contr_em_orders,axis=0) # average over all orders
            self.idx_maxcont=np.where(self.summed_contr == np.max(self.summed_contr))[0][0]
            self.params_dict['T_maxcont'] = self.model_object.temperature[self.idx_maxcont] # temperature at max emission contribution
            self.params_dict['log_P_maxcont'] = np.log10(self.pressure[self.idx_maxcont]) # pressure at max emission contribution
            self.get_ratios() # save isotope & element ratios in final params dict

            # save abundances of species at maximum emission contribution
            if self.chemistry in ['equchem','quequchem']:
                for species_i in self.species_names:
                    minus,median,plus=np.percentile(np.log10(np.array(self.VMR_dict[species_i])[:,self.idx_maxcont]), [15.9,50.0,84.1], axis=0)
                    self.params_dict[f'log_{species_i}'] = median
                    self.params_dict[f'log_{species_i}_err'] = (minus-median,plus-median)
            
            # get scaling parameters phi_ij and s2_ij of bestfit model through likelihood
            #self.log_likelihood = self.LogLike(self.model_flux0, self.Cov)
            lnL = self.PMN_lnL()
            self.params_dict['phi_ij']=self.LogLike.phi
            self.params_dict['s2_ij']=self.LogLike.s2
            if self.callback_label=='final_':
                self.params_dict['chi2']=self.LogLike.chi2_red # save reduced chi^2 of fiducial model
                self.params_dict['lnZ']=self.lnZ # save lnZ of fiducial model
                self.params_dict['lnL']=lnL
                
            if self.primary_label==False:
                self.params_dict['phi_ij_comp']=self.model_object.phi_components

            self.model_flux=np.zeros_like(self.model_flux0)
            phi_ij=self.params_dict['phi_ij']
            for order in range(self.n_orders):
                for det in range(self.n_dets):
                    self.model_flux[order,det]=phi_ij[order,det]*self.model_flux0[order,det] # scale model accordingly

            spectrum=np.full(shape=(2048*7*3,2),fill_value=np.nan)
            spectrum[:,0]=self.data_wave.flatten()
            spectrum[:,1]=self.model_flux.flatten()

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                save_pickle(self.params_dict,f'{self.output_dir}/params_dict.pickle')
                np.savetxt(f'{self.output_dir}/bestfit_spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
        
        return self.params_dict,self.model_flux

    def get_ratios(self): # can only be run after self.evaluate()

        bounds_array=[]
        for key in self.parameters.param_keys:
            bounds=self.parameters.param_priors[key]
            bounds_array.append(bounds)
        bounds_array=np.array(bounds_array)

        # one value for each parameters = one sample
        all_samples = np.empty((len(self.posterior[list(self.posterior.keys())[0]][0]),len(list(self.parameters.free_params.keys()))))
        for k,key in enumerate(list(self.parameters.free_params.keys())):
            all_samples[:,k] = self.posterior[key][0]

        temp_dist=pathlib.Path(f'{self.output_dir}/temperature_dist.npy')
        VMR_dict=pathlib.Path(f'{self.output_dir}/VMR_dict.pickle')

        # add all ratio posteriors to posterior dict, check C/O if it has been done already
        if ('C/O' in self.posterior.keys()) and temp_dist.exists() and self.chemistry=='freechem':
            self.temp_dist=np.load(temp_dist)

        elif temp_dist.exists() and VMR_dict.exists() and self.chemistry in ['equchem','quequchem']:
            self.temp_dist=np.load(temp_dist)
            self.VMR_dict= load_pickle(VMR_dict)
            
        elif self.chemistry in ['equchem','quequchem']:

            stop=10
            temperature_distribution=[] # for each of the n_atm_layers
            VMRs=[]
            for j,sample in enumerate(all_samples):
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                model_object=pRT_spectrum(self)
                temperature_distribution.append(np.array(model_object.temperature))
                VMRs.append(model_object.VMR_dict)
                # when testing from my laptop, or it takes too long to evaluate C/O, C/H, temps for all samples (22min)
                if getpass.getuser()=="natalie" and j>stop: 
                    remaining=len(all_samples)-(j+1)
                    temperature_distribution+=[self.model_object.temperature]*remaining
                    break
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)

            self.VMR_dict={}
            for molec in VMRs[0].keys():
                vmr_list=[]
                for i in range(len(self.posterior)):
                    vmr_list.append(VMRs[i][molec])
                self.VMR_dict[molec]=vmr_list # reformat to make it easier to work with

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                np.save(f'{self.output_dir}/temperature_dist.npy',self.temp_dist)
                save_pickle(self.VMR_dict,f'{self.output_dir}/VMR_dict.pickle')

        elif self.chemistry=='freechem':

            mathtext = []
            get_ratios = []
            if 'log_13CO' in self.parameters.param_keys:
                get_ratios.append(('12CO','13CO'))
                mathtext.append(r'log $^{12}$CO/$^{13}$CO')
            if 'log_C17O' in self.parameters.param_keys:
                get_ratios.append(('12CO','C17O'))
                mathtext.append(r'log $^{12}$CO/C$^{17}$O')
            if 'log_C18O' in self.parameters.param_keys:
                get_ratios.append(('12CO','C18O'))
                mathtext.append(r'log $^{12}$CO/C$^{18}$O')
            if 'log_H2(18)O' in self.parameters.param_keys:
                get_ratios.append(('H2O','H2(18)O'))
                mathtext.append(r'log H$_2$O/H$_2^{18}$O')

            for i,(m1,m2) in enumerate(get_ratios): # isotope ratios    
                p1=self.posterior[f'log_{m1}'][0]
                p2=self.posterior[f'log_{m2}'][0]
                log_ratio=p1-p2
                median,minus_err,plus_err=self.get_quantiles(log_ratio)
                self.params_dict[f'log_{m1}/{m2}']=median
                self.params_dict[f'log_{m1}/{m2}_err']=(minus_err,plus_err)
                self.posterior[f'log_{m1}/{m2}'] = (log_ratio,mathtext[i])

            CO_distribution=[]
            CH_distribution=[]
            temperature_distribution=[] # for each of the n_atm_layers
            stop=10

            for j,sample in enumerate(all_samples):
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                model_object=pRT_spectrum(self)
                CO_distribution.append(model_object.CO)
                CH_distribution.append(model_object.FeH)
                temperature_distribution.append(np.array(model_object.temperature))
                # when testing from my laptop, or it takes too long to evaluate C/O, C/H, temps for all samples (22min)
                if getpass.getuser()=="natalie" and j>stop: 
                    remaining=len(all_samples)-(j+1)
                    temperature_distribution+=[self.model_object.temperature]*remaining
                    CO_distribution+=[self.model_object.CO]*remaining
                    CH_distribution+=[self.model_object.FeH]*remaining
                    CO_distribution=np.array(CO_distribution)
                    CH_distribution=np.array(CH_distribution)
                    break
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)

            median,minus_err,plus_err=self.get_quantiles(CO_distribution)
            self.params_dict['C/O']=median
            self.params_dict['C/O_err']=(minus_err,plus_err)
            self.posterior['C/O'] = (np.array(CO_distribution),'C/O')

            median,minus_err,plus_err=self.get_quantiles(CH_distribution)
            self.params_dict['C/H']=median
            self.params_dict['C/H_err']=(minus_err,plus_err)
            self.posterior['C/H'] = (np.array(CH_distribution),'[C/H]')

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                np.save(f'{self.output_dir}/temperature_dist.npy',self.temp_dist)
                post=pathlib.Path(f'{self.output_dir}/posterior_dict.pickle') 
                save_pickle(self.posterior,post) # overwrite with ratio posteriors

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True,
                 callback_label='final_',makefigs=True):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.params_dict,self.model_flux=self.get_params_and_spectrum() # all params + scaling phi_ij + s2_ij
        if makefigs:
            if callback_label=='final_':
                figs.make_all_plots(self,only_abundances=only_abundances,only_params=only_params,split_corner=split_corner)
            else:
                figs.summary_plot(self)
        
    def cross_correlation(self,ccf_species,noiserange=100): # can only be run after evaluate()

        ccf_dict={}
        ccf_acf_dict={} # save cross-correlations and auto-correlations
        orig_params_dict=self.params_dict
        CCF_results=pathlib.Path(f'{self.output_dir}/CCF_ACF_dict.pickle')
        if CCF_results.exists():
            ccf_acf_dict=load_pickle(CCF_results)

        if isinstance(ccf_species, list)==False:
            ccf_species=[ccf_species] # if only one, make list so that it works in the for loop

        RVs=np.arange(-500,500,1) # km/s
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)

        for j,species_i in enumerate(ccf_species):

            if CCF_results.exists()==False:
                # create final model without opacity from a certain specie
                exclusion_dict=self.params_dict.copy()
                if self.chemistry=='freechem':
                    exclusion_dict[f'log_{species_i}']=-14 # exclude from model
                   
                # interpolate=False for CCF: not interpolated onto data_wave so that wl padding not cut off
                self.parameters.params=exclusion_dict
                if self.chemistry in ['equchem','quequchem']:
                    exclusion_model_object=pRT_spectrum(self,interpolate=False,leave_out=species_i)
                else:
                    exclusion_model_object=pRT_spectrum(self,interpolate=False)
                
                # shape (n_orders,length of uninterpolated wavelengths), must still be interpolated 
                exclusion_model,exclusion_model_wl=exclusion_model_object.make_spectrum()

                self.parameters.params=orig_params_dict        
                model_flux_broad,_=pRT_spectrum(self,interpolate=False).make_spectrum()

                beta=1.0-RVs/const.c.to('km/s').value
                CCF = np.zeros((self.n_orders,self.n_dets,len(RVs)))
                ACF = np.zeros((self.n_orders,self.n_dets,len(RVs))) # auto-correlation

                for order in range(self.n_orders):
                    for det in range(self.n_dets):

                        if np.isnan(self.data_flux[order,det]).all():
                            pass # skip empty order/det, CCF and ACF remains 0 

                        else:
                            wl_data=self.data_wave[order,det,self.mask_isfinite[order,det]] 
                            fl_data=self.data_flux[order,det,self.mask_isfinite[order,det]] 

                            if self.primary_label==False: # remove primary to cc only w secondary
                                fl_data-= self.model_object.primary_broadened[order,det,self.mask_isfinite[order,det]]
                            
                            wl_excl=exclusion_model_wl[order]
                            fl_excl=exclusion_model[order]*self.params_dict['phi_ij'][order,det]
                            fl_final=model_flux_broad[order]*self.params_dict['phi_ij'][order,det]

                            # data minus model without certain species
                            fl_excl_rebinned=interp1d(wl_excl,fl_excl)(wl_data) # rebin to allow subtraction
                            residuals=fl_data-fl_excl_rebinned
                            residuals-=np.nanmean(residuals) # mean should be at zero
                            self.Cov[order,det].get_cholesky() # in case it hasn't been called yet
                            cov_0_res=self.Cov[order,det].solve(residuals)
                            
                            # excluded species template: complete final model minus final model w/o species
                            species_template=fl_final-fl_excl
                            species_template_rebinned=interp1d(wl_excl,species_template)(wl_data) # rebin for Cov
                            species_template_rebinned-=np.nanmean(species_template_rebinned) # mean should be at zero
                            cov_0_temp=self.Cov[order,det].solve(species_template_rebinned)
                            wl_shift=wl_data[:, np.newaxis]*beta[np.newaxis, :]
                            template_shift=interp1d(wl_excl,species_template)(wl_shift) # interpolate template onto shifted wl
                            template_shift-=np.nanmean(template_shift) # mean should be at zero

                            CCF[order,det]=(template_shift.T).dot(cov_0_res)
                            ACF[order,det]=(template_shift.T).dot(cov_0_temp)

                CCF_sum=np.sum(np.sum(CCF,axis=0),axis=0) # sum CCF over all orders detectors
                ACF_sum=np.sum(np.sum(ACF,axis=0),axis=0)
                noise=np.std(CCF_sum[np.abs(RVs)>noiserange]) # mask out regions close to expected RV
                CCF_norm = CCF_sum/noise # get ccf map in S/N units
                ACF_norm = ACF_sum/noise

                SNR=CCF_norm[np.where(RVs==0)[0][0]]
                self.parameters.params=orig_params_dict

            else:
                CCF_norm,ACF_norm,SNR = ccf_acf_dict[species_i]

            ccf_dict[f'SNR_{species_i}']=SNR
            ccf_acf_dict[species_i]=(CCF_norm,ACF_norm,SNR)
            print(f'{species_i} S/N =',np.round(SNR,decimals=2))

        self.ccf_acf_dict = ccf_acf_dict
        figs.CCF_plot_all(self,ccf_species,noiserange=100)

        if CCF_results.exists()==False and ccf_species==self.species_names:
            save_pickle(ccf_acf_dict,CCF_results)

        self.params_dict.update(ccf_dict)
        save_pickle(self.params_dict,f'{self.output_dir}/params_dict.pickle') # overwrite with CCF SNR

        return ccf_dict

    def CCF_residuals(self,ccf_species,noiserange=100): # can only be run after evaluate()

        ccf_dict={}
        ccf_acf_dict={} # save cross-correlations and auto-correlations
        CCF_results=pathlib.Path(f'{self.output_dir}/CCF_residuals.pickle')
        if CCF_results.exists():
            ccf_acf_dict=load_pickle(CCF_results)

        if isinstance(ccf_species, list)==False:
            ccf_species=[ccf_species] # if only one, make list so that it works in the for loop

        RVs=np.arange(-500,500,1) # km/s
        for j,ccf_species_i in enumerate(ccf_species):

            if CCF_results.exists()==False or (ccf_species_i not in ccf_acf_dict):

                # create template with only selected species at equibilrium abundance
                parameters_spec = self.params_dict
                parameters_spec.update({'C/O': self.params_dict['C/O'],
                                'Fe/H': self.params_dict['C/H']})
                ratios_free,ratios_equ = get_ratios(self,equ_too=True)
                for r,e in zip(ratios_free,ratios_equ):
                    parameters_spec.update({e: self.params_dict[r]})
                parameters_spec = Parameters({}, parameters_spec)
                parameters_spec.param_priors['log_l']=[-3,0]
                retr_spec = Retrieval(target=self.target,parameters=parameters_spec, 
                                        species_names=self.species_names,Nlive=self.Nlive,
                                        evtol=self.evtol,chemistry='equchem',
                                        PT_type=self.PT_type,cloud_mode=self.cloud_mode)
                retr_spec.primary_label=True
                retr_spec.species_names = [ccf_species_i]
                retr_spec.species_pRT, retr_spec.species_hill =retr_spec.get_pRT_hill(retr_spec.species_names)
                retr_spec.atmosphere_objects = retr_spec.get_atmosphere_objects(for_species=ccf_species_i)
                template_fluxes=pRT_spectrum(retr_spec).make_spectrum()
                template_waves = self.data_wave
                
                cut = 2 # remove values on edge because they were problematic??
                beta=1.0-RVs/const.c.to('km/s').value
                CCF = np.zeros((self.n_orders,self.n_dets,len(RVs)))
                ACF = np.zeros((self.n_orders,self.n_dets,len(RVs))) # auto-correlation

                for order in range(self.n_orders):
                    for det in range(self.n_dets):

                        if np.isnan(self.data_flux[order,det]).all():
                            pass # skip empty order/det, CCF and ACF remains 0 

                        else:
                            
                            template_flux=template_fluxes[order][cut:-cut]
                            template_wl = template_waves[order][cut:-cut]
                            
                            wl_data=self.data_wave[order,det,self.mask_isfinite[order,det]]
                            fl_data = self.data_flux[order,det,self.mask_isfinite[order,det]]-self.model_flux[order,det,self.mask_isfinite[order,det]]

                            #if self.primary_label==False: # remove primary to cc only w secondary
                                #fl_data-= self.model_object.primary_broadened[order,det,self.mask_isfinite[order,det]]
                            template_flux = rem_cont(template_wl,template_flux)
                            plt.plot(template_wl,template_flux,c='tab:blue')
                            plt.plot(wl_data,fl_data,c='tab:orange')

                            fl_data-=np.nanmedian(fl_data)
                            self.Cov[order,det].get_cholesky() # in case it hasn't been called yet
                            cov_0_data=self.Cov[order,det].solve(fl_data)                            
                            wl_shift=wl_data[:, np.newaxis]*beta[np.newaxis, :]
                            template_shift=interp1d(template_wl,template_flux)(wl_shift) # interpolate template onto shifted wl
                            #template_shift-= np.nanmedian(template_shift)  
                            template_shift = np.array([template_shift[:,i] - np.nanmedian(template_shift[:,i]) for i in range(template_shift.shape[1])]).T
                            #print(order,det, np.nanmedian(template_shift),np.nanmedian(fl_data))
                            cov_0_temp=self.Cov[order,det].solve(template_shift[:,0])
                            CCF[order,det]=(template_shift.T).dot(cov_0_data)
                            ACF[order,det]=(template_shift.T).dot(cov_0_temp)

                plt.savefig(f'./xccf/{ccf_species_i}.png')
                plt.close()
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

            else:
                CCF_norm,ACF_norm,SNR = ccf_acf_dict[ccf_species_i]

            ccf_dict[f'SNR_{ccf_species_i}']=SNR
            ccf_acf_dict[ccf_species_i]=(CCF_norm,ACF_norm,SNR)
            print(f'{ccf_species_i} S/N =',np.round(SNR,decimals=2))

        self.ccf_acf_dict = ccf_acf_dict
        figs.CCF_plot_all(self,ccf_species,noiserange=100,show_ACF=True)

        if CCF_results.exists()==False and ccf_species==self.species_names:
            save_pickle(ccf_acf_dict,CCF_results)

        self.params_dict.update(ccf_dict)
        #save_pickle(self.params_dict,f'{self.output_dir}/params_dict.pickle')

        return ccf_dict

    def bayes_evidence(self,bayes_species,evidence_dict,retrieval_output_dir):

        bayes_dict=evidence_dict
        self.output_dir=pathlib.Path(f'{self.output_dir}/evidence_retrievals') # store output in separate folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print('\n ----------------- Current bayes_dict= ----------------- \n',bayes_dict)

        if isinstance(bayes_species, list)==False:
            bayes_species=[bayes_species] # if only one, make list so that it works in for loop

        for species_i in bayes_species: # exclude from retrieval

            self.prefix=f'pmn_wo{species_i}_' 
            finish=pathlib.Path(f'{self.output_dir}/final_wo{species_i}_posterior_dict.pickle')
            if finish.exists():
                print(f'\n ----------------- Evidence retrieval for {species_i} already done ----------------- \n')
                setback_prior=False
            else:
                print(f'\n ----------------- Starting evidence retrieval for {species_i} ----------------- \n')
                setback_prior=True
                if self.chemistry=='freechem':
                    original_prior=self.parameters.param_priors[f'log_{species_i}']
                    self.parameters.param_priors[f'log_{species_i}']=[-15,-14] # exclude from retrieval
                elif self.chemistry in ['equchem','quequchem']:
                    if species_i=='13CO':
                        key='log_C12_1S3_ratio'
                    elif species_i=='H2(18)O':
                        key='log_O16_18_ratio'
                    original_prior=self.parameters.param_priors[key]
                    self.parameters.param_priors[key]=[14,15] # exclude from retrieval

                self.callback_label=f'live_wo{species_i}_'
                self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance,resume=True)
            
            self.callback_label=f'final_wo{species_i}_'
            self.evaluate(callback_label=self.callback_label) # gets self.lnZ_ex
            ex_model=pRT_spectrum(self).make_spectrum()      
            lnL = self.LogLike(ex_model, self.Cov) # call function to generate chi2
            chi2_ex = self.LogLike.chi2_red # reduced chi^2
            lnB,sigma=self.compare_evidence(self.lnZ, self.lnZ_ex)
            print(f'sigma_{species_i}=',sigma)
            bayes_dict[f'lnBm_{species_i}']=lnB
            bayes_dict[f'sigma_{species_i}']=sigma
            bayes_dict[f'chi2_wo_{species_i}']=chi2_ex  
            save_pickle(bayes_dict,f'{retrieval_output_dir}/evidence_dict.pickle') # save results at each step

            # set back param priors for next retrieval
            if setback_prior==True:
                if self.chemistry=='freechem':
                    self.parameters.param_priors[f'log_{species_i}']=original_prior 
                elif self.chemistry in ['equchem','quequchem']:
                    if species_i=='13CO':
                        key='log_C12_13_ratio'
                    elif species_i=='H2(18)O':
                        key='log_O16_18_ratio'
                    self.parameters.param_priors[key]=original_prior
            
        return bayes_dict

    def compare_evidence(self,ln_Z_A,ln_Z_B):
        '''
        Convert log-evidences of two models to a sigma confidence level
        Originally from Benneke & Seager (2013), adapted from samderegt/retrieval_base
        '''

        from scipy.special import lambertw as W
        from scipy.special import erfcinv

        ln_B = ln_Z_A-ln_Z_B
        sign=1
        if ln_B<0: # ln_Z_B larger -> second model favored
            sign=-1
            ln_B*=sign # can't handle negative values (-> nan), multiply back later
        try:
            p = np.real(np.exp(W((-1.0/(np.exp(ln_B)*np.exp(1))),-1)))
            sigma = np.sqrt(2)*erfcinv(p)
        except RuntimeWarning:
            sigma=np.inf 
        return ln_B*sign,sigma*sign

    def run_retrieval(self,bayes_species=None): 

        retrieval_output_dir=self.output_dir # save end results here

        print(f'\n ------ {self.target.name} - {self.chemistry} - {self.PT_type} - Nlive: {self.Nlive} - ev: {self.evtol} ------ \n')

        # run main retrieval if hasn't been run yet, else skip to cross-corr and bayes
        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists()==False:
            print('\n ----------------- Starting main retrieval. ----------------- \n')
            self.PMN_run(N_live_points=self.Nlive,evidence_tolerance=self.evtol)
        else:
            print('\n ----------------- Main retrieval exists. ----------------- \n')
        self.evaluate() # created and saves self.params_dict

        ccf_dict=self.cross_correlation(self.species_names) # cross-corr all species
        self.params_dict.update(ccf_dict)
        save_pickle(self.params_dict,f'{retrieval_output_dir}/params_dict.pickle') # overwrite with added CCF SNR
    
        print('Parameters:\n',self.params_dict)
        if bayes_species!=None:
            evidence_dict=pathlib.Path(f'{retrieval_output_dir}/evidence_dict.pickle')
            if evidence_dict.exists()==False: # to avoid overwriting sigmas from other evidence retrievals
                print('\n ----------------- Creating evidence dict ----------------- \n')
                self.evidence_dict={}
            else:
                print('\n ----------------- Continuing existing evidence dict ----------------- \n')
                self.evidence_dict= load_pickle(evidence_dict)

            bayes_dict=self.bayes_evidence(bayes_species,evidence_dict=self.evidence_dict,retrieval_output_dir=retrieval_output_dir)
            print('\n ----------------- Final evidence dict ----------------- \n',bayes_dict)
            save_pickle(bayes_dict,f'{retrieval_output_dir}/evidence_dict.pickle') # save new results in separate dict

        output_file=pathlib.Path('retrieval.out')
        if output_file.exists():
            os.system(f"mv {output_file} {retrieval_output_dir}")

        print('\n ----------------- Done ---------------- \n')

        
        

