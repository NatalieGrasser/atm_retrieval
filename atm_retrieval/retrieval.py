import getpass
import os
if getpass.getuser() == "grasser": # when runnig from LEM
    from atm_retrieval.pRT_model import pRT_spectrum
    import atm_retrieval.figures as figs
    from atm_retrieval.covariance import *
    from atm_retrieval.log_likelihood import *
elif getpass.getuser() == "natalie": # when testing from my laptop
    from pRT_model import pRT_spectrum
    import figures as figs
    from covariance import *
    from log_likelihood import *

import numpy as np
import pymultinest
import pathlib
import pickle
from petitRADTRANS import Radtrans
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.interpolate import interp1d
import copy
#import warnings
#warnings.filterwarnings("ignore", category=np.linalg.LinAlgError) 


class Retrieval:

    def __init__(self,target,parameters,output_name,chemistry='freechem',
                 GP=True,cloud_mode='gray',PT_type='PTknot',redo=False):

        self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])
        
        self.target=target
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.mask_isfinite=target.get_mask_isfinite() # mask nans, shape (orders,detectors)
        self.separation,self.err_eff=target.prepare_for_covariance()
        self.parameters=parameters
        self.chemistry=chemistry # freechem/equchem
        self.species=self.get_species(param_dict=self.parameters.params)

        self.n_orders, self.n_dets, _ = self.data_flux.shape # shape (orders,detectors,pixels)
        self.n_params = len(parameters.free_params)
        self.output_name=output_name
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
    
        self.LogLike = LogLikelihood(retrieval_object=self,scale_flux=True,scale_err=True)

        # load atmosphere objects here and not in likelihood/pRT_model to make it faster
        # redo atmosphere objects when introdocuing new species or MgSiO3 clouds
        self.atmosphere_objects=self.get_atmosphere_objects(redo=redo)
        self.callback_label='live_' # label for plots
        self.prefix='pmn_'

        # will be updated, but needed as None until then
        self.bestfit_params=None 
        self.posterior = None
        self.final_params=None
        if self.target.name in ['2M0355','test']:
            self.color1='deepskyblue' # color of retrieval output
            self.color2='tab:blue' # color of residuals
        elif self.target.name=='2M1425':
            #self.color1='limegreen' # color of retrieval output
            #self.color2='forestgreen' # color of residuals
            self.color1='lightcoral' # color of retrieval output
            self.color2='indianred' # color of residuals

    def get_species(self,param_dict): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if self.chemistry=='freechem':
            self.chem_species=[]
            for par in param_dict:
                if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
                    if par in ['log_g','log_Kzz','log_P_base_gray','log_opa_base_gray','log_a','log_l']: # skip
                        pass
                    else:
                        self.chem_species.append(par)
            species=[]
            for chemspec in self.chem_species:
                species.append(species_info.loc[chemspec[4:],'pRT_name'])
        elif self.chemistry=='equchem':
            self.chem_species=['H2O','12CO','13CO','C18O','C17O','CH4','NH3',
                         'HCN','H2(18)O','H2S','CO2','HF','OH'] # HF, OH not in pRT chem equ table
            species=[]
            for chemspec in self.chem_species:
                species.append(species_info.loc[chemspec,'pRT_name'])
        return species

    def get_atmosphere_objects(self,redo=False,broader=True):

        atmosphere_objects=[]
        file=pathlib.Path(f'atmosphere_objects.pickle')
        if file.exists() and redo==False:
            with open(file,'rb') as file:
                atmosphere_objects=pickle.load(file)
                return atmosphere_objects
        else:
            for order in range(self.n_orders):
                wl_pad=7 # wavelength padding because spectrum is not wavelength shifted yet
                if broader==True:  # larger wl pad needed when shifting during cross-correlation
                    rv_max = 501 # maximum RV for cross-corr
                    wl_max= np.max(self.K2166)
                    wl_pad = 1.1*rv_max/(const.c.to('km/s').value)*wl_max
                wlmin=np.min(self.K2166[order])-wl_pad
                wlmax=np.max(self.K2166[order])+wl_pad
                wlen_range=np.array([wlmin,wlmax])*1e-3 # nm to microns

                atmosphere = Radtrans(line_species=self.species,
                                    rayleigh_species = ['H2', 'He'],
                                    continuum_opacities = ['H2-H2', 'H2-He'],
                                    wlen_bords_micron=wlen_range, 
                                    mode='lbl',
                                    cloud_species=self.cloud_species,
                                    do_scat_emis=self.do_scat_emis,
                                    lbl_opacity_sampling=self.lbl_opacity_sampling) # take every nth point (=3 in deRegt+2024)
                
                atmosphere.setup_opa_structure(self.pressure)
                atmosphere_objects.append(atmosphere)
            with open(file,'wb') as file:
                pickle.dump(atmosphere_objects,file)
            return atmosphere_objects

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(parameters=self.parameters.params,
                                     data_wave=self.data_wave,
                                     target=self.target,
                                     atmosphere_objects=self.atmosphere_objects,
                                     species=self.species,
                                     chemistry=self.chemistry,
                                     cloud_mode=self.cloud_mode,
                                     PT_type=self.PT_type)
        self.model_flux=self.model_object.make_spectrum()
        for j in range(self.n_orders): # update covariance matrix
            for k in range(self.n_dets):
                if not self.mask_isfinite[j,k].any(): # skip empty order/detector
                    continue
                self.Cov[j,k](self.parameters.params)
        if False: # debugging
            ord=4
            det=2
            plt.plot(self.data_wave[ord,det],self.data_flux[ord,det],lw=0.8)
            plt.plot(self.data_wave[ord,det],self.model_flux[ord,det],alpha=0.7,lw=0.8)
        ln_L = self.LogLike(self.model_flux, self.Cov) # retrieve log-likelihood
        return ln_L

    def PMN_run(self,N_live_points=400,evidence_tolerance=0.5,resume=False):
        pymultinest.run(LogLikelihood=self.PMN_lnL,Prior=self.parameters,n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'{self.output_dir}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=True, sampling_efficiency = 0.5,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # default is 0.5, high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=100)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        #self.callback_label='live_' # label for plots
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        np.save(f'{self.output_dir}/{self.callback_label}bestfit_params.npy',self.bestfit_params)
        self.posterior = posterior[:,:-2] # remove last 2 columns
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.final_params,self.final_spectrum=self.get_final_params_and_spectrum()
        figs.summary_plot(self)
     
    def PMN_analyse(self):
        analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                        outputfiles_basename=f'{self.output_dir}/{self.prefix}')  # set up analyzer object
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
        self.posterior = self.posterior[:,:-1] # shape 
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # read params of best-fitting model, highest likelihood
        if self.prefix=='pmn_':
            self.lnZ = stats['nested importance sampling global log-evidence']
        else: # when doing exclusion retrievals
            self.lnZ_ex = stats['nested importance sampling global log-evidence']

    def get_quantiles(self,posterior):
        quantiles = np.array([np.percentile(posterior[:,i], [16.0,50.0,84.0], axis=-1) for i in range(posterior.shape[1])])
        medians=quantiles[:,1] # median of all params
        plus_err=quantiles[:,2]-medians # +error
        minus_err=quantiles[:,0]-medians # -error
        return medians,minus_err,plus_err

    def get_final_params_and_spectrum(self,contribution=True,save=False): 
        
        # make dict of constant params + evaluated params + their errors
        self.final_params=self.parameters.constant_params.copy() # initialize dict with constant params
        medians,minus_err,plus_err=self.get_quantiles(self.posterior)

        for i,key in enumerate(self.parameters.param_keys):
            self.final_params[key]=medians[i] # add median of evaluated params (more robust)
            self.final_params[f'{key}_err']=(minus_err[i],plus_err[i]) # add errors of evaluated params
            self.final_params[f'{key}_bf']=self.bestfit_params[i] # bestfit params with highest lnL (can differ from median, not as robust)

        # create final spectrum
        self.final_object=pRT_spectrum(parameters=self.final_params,data_wave=self.data_wave,
                                       target=self.target,species=self.species,
                                       atmosphere_objects=self.atmosphere_objects,
                                       chemistry=self.chemistry,contribution=contribution,
                                       PT_type=self.PT_type)
        self.final_model=self.final_object.make_spectrum()

        # get isotope and element ratios and save them in final params dict
        self.get_ratios()

        # get scaling parameters phi_ij and s2_ij of bestfit model through likelihood
        self.log_likelihood = self.LogLike(self.final_model, self.Cov)
        self.final_params['phi_ij']=self.LogLike.phi
        self.final_params['s2_ij']=self.LogLike.s2
        if self.callback_label=='final_':
            self.final_params['chi2']=self.LogLike.chi2_0_red # save reduced chi^2 of fiducial model
            self.final_params['lnZ']=self.lnZ # save lnZ of fiducial model

        if save==True:
            with open(f'{self.output_dir}/{self.callback_label}params_dict.pickle','wb') as file:
                pickle.dump(self.final_params,file)
            
        self.final_spectrum=np.zeros_like(self.final_model)
        phi_ij=self.final_params['phi_ij']
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                self.final_spectrum[order,det]=phi_ij[order,det]*self.final_model[order,det] # scale model accordingly
        
        spectrum=np.full(shape=(2048*7*3,2),fill_value=np.nan)
        spectrum[:,0]=self.data_wave.flatten()
        spectrum[:,1]=self.final_spectrum.flatten()
        np.savetxt(f'{self.output_dir}/{self.callback_label}spectrum.txt',spectrum,delimiter=' ',header='wavelength (nm) flux')
        
        return self.final_params,self.final_spectrum

    def get_ratios(self,output=False): # can only be run after self.evaluate()
        if self.chemistry=='equchem':
            C1213=1/self.final_params['C13_12_ratio']
            O1618=1/self.final_params['O18_16_ratio']
            O1617=1/self.final_params['O17_16_ratio']
            FeH=self.final_params['Fe/H']
            CO=self.final_params['C/O']
            if output:
                return C1213,O1618,O1617,FeH,CO

        if self.chemistry=='freechem':
            log_12CO=self.final_params['log_12CO']
            log_12CO_me=(self.final_params['log_12CO_err'][0]) # minus error
            log_12CO_pe=(self.final_params['log_12CO_err'][1]) # plus error

            log_13CO=self.final_params['log_13CO']
            log_13CO_me=(self.final_params['log_13CO_err'][0]) # minus error
            log_13CO_pe=(self.final_params['log_13CO_err'][1]) # plus error

            log_C17O=self.final_params['log_C17O']
            log_C17O_me=(self.final_params['log_C17O_err'][0]) # minus error
            log_C17O_pe=(self.final_params['log_C17O_err'][1]) # plus error

            log_C18O=self.final_params['log_C18O']
            log_C18O_me=(self.final_params['log_C18O_err'][0]) # minus error
            log_C18O_pe=(self.final_params['log_C18O_err'][1]) # plus error

            log_H2O=self.final_params['log_H2O']
            log_H2O_me=(self.final_params['log_H2O_err'][0]) # minus error
            log_H2O_pe=(self.final_params['log_H2O_err'][1]) # plus error

            log_H218O=self.final_params['log_H2(18)O']
            log_H218O_me=(self.final_params['log_H2(18)O_err'][0]) # minus error
            log_H218O_pe=(self.final_params['log_H2(18)O_err'][1]) # plus error

            def error_prop(f,A,Ame,Ape,B,Bme,Bpe):
                fme=np.sqrt(f**2*((Ame/A)**2+(Bme/B)**2)) # minus error of f when f=A*B
                fpe=np.sqrt(f**2*((Ape/A)**2+(Bpe/B)**2)) # plus error of f when f=A*B
                return fme,fpe
            
            self.final_params['12CO/13CO']=10**(log_12CO-log_13CO)
            self.final_params['12CO/13CO_err']=error_prop(log_12CO-log_13CO,log_12CO,log_12CO_me,log_12CO_pe,
                                                          log_13CO,log_13CO_me,log_13CO_pe)
            self.final_params['12CO/C18O']=10**(log_12CO-log_C18O)
            self.final_params['12CO/C18O_err']=error_prop(log_12CO-log_C18O,log_12CO,log_12CO_me,log_12CO_pe,
                                                          log_C18O,log_C18O_me,log_C18O_pe)
            self.final_params['12CO/C17O']=10**(log_12CO-log_C17O)
            self.final_params['12CO/C17O_err']=error_prop(log_12CO-log_C17O,log_12CO,log_12CO_me,log_12CO_pe,
                                                          log_C17O,log_C17O_me,log_C17O_pe)
            self.final_params['H2O/H2(18)O']=10**(log_H2O-log_H218O)
            self.final_params['H2O/H2(18)O_err']=error_prop(log_H2O-log_H218O,log_H2O,log_H2O_me,log_H2O_pe,
                                                          log_H218O,log_H218O_me,log_H218O_pe)
            self.final_params['Fe/H']=self.final_object.FeH
            self.final_params['C/O']=self.final_object.CO

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True,
                 callback_label='final_',save=False):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.final_params,self.final_spectrum=self.get_final_params_and_spectrum(save=save) # all params: constant + free + scaling phi_ij + s2_ij
        if callback_label=='final_':
            #figs.plot_spectrum_split(self)
            #figs.plot_spectrum_inset(self)
            #figs.plot_pt(self)
            #figs.summary_plot(self)
            figs.make_all_plots(self,only_abundances=only_abundances,only_params=only_params,split_corner=split_corner)
        else:
            figs.summary_plot(self)
        
    def cross_correlation(self,molecules,noiserange=50): # can only be run after evaluate()

        ccf_dict={}
        CCF_list=[]
        ACF_list=[]
        if isinstance(molecules, list)==False:
            molecules=[molecules] # if only one, make list so that it works in for loop

        for molecule in molecules:
            # create final model without opacity from a certain molecule
            exclusion_dict=self.final_params.copy()
            exclusion_dict[f'log_{molecule}']=-12 # exclude molecule from model

            # necessary for cross-correlation:
            # interpolate=False: not interpolated onto data_wave so that wl padding not cut off
            # exclusion_model shape (n_orders,length of uninterpolated wavelengths)
            # must still be shaped correctly and interpolated
            exclusion_model,exclusion_model_wl=pRT_spectrum(parameters=exclusion_dict,
                                        data_wave=self.data_wave,
                                        target=self.target,species=self.species,
                                        atmosphere_objects=self.atmosphere_objects,
                                        chemistry=self.chemistry,PT_type=self.PT_type,
                                        interpolate=False).make_spectrum()
            
            final_model_broad,_=pRT_spectrum(parameters=self.final_params,
                                        data_wave=self.data_wave,
                                        target=self.target,species=self.species,
                                        atmosphere_objects=self.atmosphere_objects,
                                        chemistry=self.chemistry,PT_type=self.PT_type,
                                        interpolate=False).make_spectrum()

            RVs=np.arange(-500,500,1) # km/s
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
                        
                        wl_excl=exclusion_model_wl[order]
                        fl_excl=exclusion_model[order]*self.final_params['phi_ij'][order,det]
                        fl_final=final_model_broad[order]*self.final_params['phi_ij'][order,det]

                        # data minus model without certain molecule
                        fl_excl_rebinned=interp1d(wl_excl,fl_excl)(wl_data) # rebin to allow subtraction
                        residuals=fl_data-fl_excl_rebinned
                        residuals-=np.nanmean(residuals) # mean should be at zero
                        self.Cov[order,det].get_cholesky() # in case it hasn't been called yet
                        cov_0_res=self.Cov[order,det].solve(residuals)
                        
                        # excluded molecule template: complete final model minus final model w/o molecule
                        molecule_template=fl_final-fl_excl
                        molecule_template_rebinned=interp1d(wl_excl,molecule_template)(wl_data) # rebin for Cov
                        molecule_template_rebinned-=np.nanmean(molecule_template_rebinned) # mean should be at zero
                        cov_0_temp=self.Cov[order,det].solve(molecule_template_rebinned)
                        wl_shift=wl_data[:, np.newaxis]*beta[np.newaxis, :]
                        template_shift=interp1d(wl_excl,molecule_template)(wl_shift) # interpolate template onto shifted wl
                        template_shift-=np.nanmean(template_shift) # mean should be at zero

                        CCF[order,det]=(template_shift.T).dot(cov_0_res)
                        ACF[order,det]=(template_shift.T).dot(cov_0_temp)

            CCF_sum=np.sum(np.sum(CCF,axis=0),axis=0) # sum CCF over all orders detectors
            ACF_sum=np.sum(np.sum(ACF,axis=0),axis=0)
            noise=np.std(CCF_sum[np.abs(RVs)>noiserange]) # mask out regions close to expected RV
            CCF_norm = CCF_sum/noise # get ccf map in S/N units
            ACF_norm = ACF_sum/noise
            SNR=CCF_norm[np.where(RVs==0)[0][0]]
            CCF_list.append(CCF_norm)
            ACF_list.append(ACF_norm)
            ccf_dict[f'SNR_{molecule}']=SNR
            #self.final_params[f'SNR_{molecule}']=SNR
            #print('self.final_params=\n',self.final_params)   
            figs.CCF_plot(self,molecule,RVs,CCF_norm,ACF_norm,noiserange=noiserange)
        self.CCF_list=CCF_list
        self.ACF_list=ACF_list
        return ccf_dict

    def bayes_evidence(self,molecules):

        bayes_dict={}
        self.output_dir=pathlib.Path(f'{self.output_dir}/evidence_retrievals') # store output in separate folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        old_parameters=copy.copy(self.parameters) # keep, self.params must be overwritten for other functions

        if isinstance(molecules, list)==False:
            molecules=[molecules] # if only one, make list so that it works in for loop

        for molecule in molecules: # exclude molecule from retrieval
            self.parameters=copy.copy(old_parameters)
            self.parameters.param_priors[f'log_{molecule}']=[-12,-12] # exclude molecule from retrieval
            #self.parameters.params[f'log_{molecule}']=-12 # exclude molecule from retrieval
            #key=f'log_{molecule}'
            #if key in self.parameters.params: del self.parameters.params[key]
            #print('New parameter priors for exclusion retrieval:\n',self.parameters.params)
            self.callback_label=f'live_wo{molecule}_'
            self.prefix=f'pmn_wo{molecule}_' 
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance)
            self.callback_label=f'final_wo{molecule}_'
            self.evaluate(callback_label=self.callback_label)
            self.PMN_analyse() # gets self.lnZ_ex
            print(f'lnZ=',self.lnZ)
            print(f'lnZ_{molecule}=',self.lnZ_ex)
            lnB,sigma=self.compare_evidence(self.lnZ, self.lnZ_ex)
            print(f'lnBm_{molecule}=',lnB)
            print(f'sigma_{molecule}=',sigma)
            bayes_dict[f'lnBm_{molecule}']=lnB
            bayes_dict[f'sigma_{molecule}']=sigma
            #self.final_params[f'lnBm_{molecule}']=lnB
            #self.final_params[f'sigma_{molecule}']=sigma # save result in dict         
            #print('self.final_params=\n',self.final_params) 
            print('bayes_dict=',bayes_dict)  
        return bayes_dict

    def compare_evidence(self,ln_Z_A,ln_Z_B):
        '''
        Convert log-evidences of two models to a sigma confidence level
        Originally from Benneke & Seager (2013)
        Adapted from samderegt/retrieval_base
        '''

        from scipy.special import lambertw as W
        from scipy.special import erfcinv

        ln_B = ln_Z_A-ln_Z_B
        sign=1
        if ln_B<0: # ln_Z_B larger -> second model favored
            sign=-1
            ln_B*=sign # can't handle negative values (-> nan), multiply back later
        p = np.real(np.exp(W((-1.0/(np.exp(ln_B)*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)
        return ln_B*sign,sigma*sign

    def run_retrieval(self,N_live_points=400,evidence_tolerance=0.5,
                      crosscorr_molecules=None,bayes_molecules=None): 
        self.N_live_points=N_live_points
        self.evidence_tolerance=evidence_tolerance
        retrieval_output_dir=self.output_dir # save end results here

        # run main retrieval if hasn't been run yet, else skip to cross-corr and bayes
        final_dict=pathlib.Path(f'{self.output_dir}/final_params_dict.pickle')
        if final_dict.exists()==False:
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance)
            save=True
        else:
            save=False
            with open(final_dict,'rb') as file:
                self.final_params=pickle.load(file) 
        self.evaluate(save=save)
        if crosscorr_molecules!=None:
            ccf_dict=self.cross_correlation(crosscorr_molecules)
            self.final_params.update(ccf_dict)
            print('self.final_params.update(ccf_dict)=\n',self.final_params)
        if bayes_molecules!=None:
            bayes_dict=self.bayes_evidence(bayes_molecules)
            self.final_params.update(bayes_dict)
            print('self.final_params.update(bayes_dict)=\n',self.final_params)

        with open(f'{retrieval_output_dir}/final_params_dict.pickle','wb') as file: # overwrite with new results
            pickle.dump(self.final_params,file)
        

