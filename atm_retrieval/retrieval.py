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
import astropy.constants as const
from scipy.interpolate import interp1d
#import warnings
#warnings.filterwarnings("ignore", category=np.linalg.LinAlgError) 


class Retrieval:

    def __init__(self,target,parameters,output_name,chemistry='freechem',
                 GP=True,cloud_mode='gray',PT_type='PTgrad',redo=False):
        
        self.target=target
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.mask_isfinite=target.get_mask_isfinite() # mask nans, shape (orders,detectors)
        self.separation,self.err_eff=target.prepare_for_covariance()
        self.K2166=target.K2166
        self.parameters=parameters
        self.chemistry=chemistry # freechem/equchem/quequchem
        self.species=self.get_species(param_dict=self.parameters.params,chemistry=self.chemistry)

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
        self.params_dict=None
        self.calc_errors=False

        self.color1=target.color1
        self.color2=target.color2

    def get_species(self,param_dict,chemistry): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if chemistry=='freechem':
            self.chem_species=[]
            for par in param_dict:
                if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
                    if par in ['log_g','log_Kzz','log_P_base_gray','log_opa_base_gray','log_a','log_l',
                               'log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio',
                               'log_Pqu_CO_CH4','log_Pqu_NH3','log_Pqu_HCN']: # skip
                        pass
                    else:
                        self.chem_species.append(par)
            species=[]
            for chemspec in self.chem_species:
                species.append(species_info.loc[chemspec[4:],'pRT_name'])
        elif chemistry in ['equchem','quequchem']:
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
        self.model_object=pRT_spectrum(self)
        
        # pass exit through a self.thing attibute and not kwarg, or pmn will be confused
        if self.calc_errors==True: # only for calc errors on Fe/H, C/O, temperatures
            return
        
        self.model_flux=self.model_object.make_spectrum()
        for j in range(self.n_orders): # update covariance matrix
            for k in range(self.n_dets):
                if not self.mask_isfinite[j,k].any(): # skip empty order/detector
                    continue
                self.Cov[j,k](self.parameters.params)
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
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        self.posterior = posterior[:,:-2] # remove last 2 columns
        self.params_dict,self.model_flux=self.get_params_and_spectrum()
        figs.summary_plot(self)
        if self.chemistry in ['equchem','quequchem']:
            figs.VMR_plot(self)
     
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

    def get_quantiles(self,posterior,flat=False):
        if flat==False: # input entire posterior of all retrieved parameters
            quantiles = np.array([np.percentile(posterior[:,i], [16.0,50.0,84.0], axis=-1) for i in range(posterior.shape[1])])
            medians=quantiles[:,1] # median of all params
            plus_err=quantiles[:,2]-medians # +error
            minus_err=quantiles[:,0]-medians # -error
        else: # input only one posterior
            quantiles = np.array([np.percentile(posterior, [16.0,50.0,84.0])])
            medians=quantiles[:,1] # median
            plus_err=quantiles[:,2]-medians # +error
            minus_err=quantiles[:,0]-medians # -error
        return medians,minus_err,plus_err

    def get_params_and_spectrum(self,save=False): 
        
        # make dict of constant params + evaluated params + their errors
        self.params_dict=self.parameters.constant_params.copy() # initialize dict with constant params
        medians,minus_err,plus_err=self.get_quantiles(self.posterior)

        for i,key in enumerate(self.parameters.param_keys):
            self.params_dict[key]=medians[i] # add median of evaluated params (more robust)
            self.params_dict[f'{key}_err']=(minus_err[i],plus_err[i]) # add errors of evaluated params
            self.params_dict[f'{key}_bf']=self.bestfit_params[i] # bestfit params with highest lnL (can differ from median, not as robust)

        # create final spectrum
        self.model_object=pRT_spectrum(self)
        model_flux0=self.model_object.make_spectrum()
        
        # get isotope and element ratios and save them in final params dict
        self.get_ratios()

        # get scaling parameters phi_ij and s2_ij of bestfit model through likelihood
        self.log_likelihood = self.LogLike(self.model_flux, self.Cov)
        self.params_dict['phi_ij']=self.LogLike.phi
        self.params_dict['s2_ij']=self.LogLike.s2
        if self.callback_label=='final_':
            self.params_dict['chi2']=self.LogLike.chi2_0_red # save reduced chi^2 of fiducial model
            self.params_dict['lnZ']=self.lnZ # save lnZ of fiducial model

        self.model_flux=np.zeros_like(model_flux0)
        phi_ij=self.params_dict['phi_ij']
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                self.model_flux[order,det]=phi_ij[order,det]*model_flux0[order,det] # scale model accordingly
        
        spectrum=np.full(shape=(2048*7*3,2),fill_value=np.nan)
        spectrum[:,0]=self.data_wave.flatten()
        spectrum[:,1]=self.model_flux.flatten()

        if save==True:
            with open(f'{self.output_dir}/{self.callback_label}params_dict.pickle','wb') as file:
                pickle.dump(self.params_dict,file)
            np.savetxt(f'{self.output_dir}/{self.callback_label}spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
        
        return self.params_dict,self.model_flux

    def get_ratios(self): # can only be run after self.evaluate()

        bounds_array=[]
        for key in self.parameters.param_keys:
            bounds=self.parameters.param_priors[key]
            bounds_array.append(bounds)
        bounds_array=np.array(bounds_array)
        
        if self.chemistry in ['equchem','quequchem']:

            for ratio in ['C/O','Fe/H','log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio']:
                p=self.posterior[:,list(self.parameters.params).index(f'{ratio}')]
                if 'ratios_posterior' in locals():
                    ratios_posterior=np.vstack([ratios_posterior,p])
                else:
                    ratios_posterior=p
            self.ratios_posterior=ratios_posterior.T
            del ratios_posterior # or in locals() won't work when loading another retrieval

            self.calc_errors=True
            temperature_distribution=[] # for each of the n_atm_layers
            x=0
            for sample in self.posterior:
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                self.PMN_lnL()
                temperature_distribution.append(self.model_object.temperature)
                x+=1
                if getpass.getuser()=="natalie" and x>20: # when testing from my laptop, or it takes too long (22min)
                    break
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)
            self.calc_errors=False # set back to False when finished

        elif self.chemistry=='freechem':
            for m1,m2 in [['12CO','13CO'],['12CO','C17O'],['12CO','C18O'],['H2O','H2(18)O']]: # isotope ratios    
                p1=self.posterior[:,list(self.parameters.params).index(f'log_{m1}')]
                p2=self.posterior[:,list(self.parameters.params).index(f'log_{m2}')]
                log_ratio=p1-p2
                median,minus_err,plus_err=self.get_quantiles(log_ratio,flat=True)
                self.params_dict[f'log_{m1}/{m2}']=median
                self.params_dict[f'log_{m1}/{m2}_err']=(minus_err,plus_err)
                if 'ratios_posterior' in locals():
                    ratios_posterior=np.vstack([ratios_posterior,log_ratio])
                else:
                    ratios_posterior=log_ratio
            self.ratios_posterior=ratios_posterior.T

            self.calc_errors=True
            CO_distribution=np.full(self.posterior.shape[0],fill_value=0.0)
            CH_distribution=np.full(self.posterior.shape[0],fill_value=0.0)
            temperature_distribution=[] # for each of the n_atm_layers
            x=0
            for j,sample in enumerate(self.posterior):
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                self.PMN_lnL()
                CO_distribution[j]=self.model_object.CO
                CH_distribution[j]=self.model_object.FeH
                temperature_distribution.append(self.model_object.temperature)
                x+=1
                if getpass.getuser()=="natalie" and x>20: # when testing from my laptop, or it takes too long (22min)
                    break
            self.CO_CH_dist=np.vstack([CO_distribution,CH_distribution]).T
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)
            self.ratios_posterior=np.hstack([self.CO_CH_dist,self.ratios_posterior])
            self.calc_errors=False # set back to False when finished

            median,minus_err,plus_err=self.get_quantiles(CO_distribution,flat=True)
            self.params_dict['C/O']=median
            self.params_dict['C/O_err']=(minus_err,plus_err)

            median,minus_err,plus_err=self.get_quantiles(CH_distribution,flat=True)
            self.params_dict['C/H']=median
            self.params_dict['C/H_err']=(minus_err,plus_err)

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True,
                 callback_label='final_',save=False,makefigs=True):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.params_dict,self.model_flux=self.get_params_and_spectrum(save=save) # all params: constant + free + scaling phi_ij + s2_ij
        if makefigs:
            if callback_label=='final_':
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
            exclusion_dict=self.params_dict.copy()
            if self.chemistry=='freechem':
                exclusion_dict[f'log_{molecule}']=-14 # exclude molecule from model
            elif self.chemistry in ['equchem','quequchem']:
                if molecule=='13CO':
                    exclusion_dict['log_C12_13_ratio']=14 # exclude molecule from model
                elif molecule=='H2(18)O':
                    exclusion_dict['log_O16_18_ratio']=14 # exclude molecule from model
                else:
                    continue

            # necessary for cross-correlation:
            # interpolate=False: not interpolated onto data_wave so that wl padding not cut off
            # exclusion_model shape (n_orders,length of uninterpolated wavelengths)
            # must still be shaped correctly and interpolated
            exclusion_model,exclusion_model_wl=pRT_spectrum(self,interpolate=False).make_spectrum()
            
            model_flux_broad,_=pRT_spectrum(self,interpolate=False).make_spectrum()

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
                        fl_excl=exclusion_model[order]*self.params_dict['phi_ij'][order,det]
                        fl_final=model_flux_broad[order]*self.params_dict['phi_ij'][order,det]

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
            #noise=np.std((CCF_sum-ACF_sum)[np.abs(RVs)>noiserange]) # mask out regions close to expected RV
            CCF_norm = CCF_sum/noise # get ccf map in S/N units
            ACF_norm = ACF_sum/noise
            SNR=CCF_norm[np.where(RVs==0)[0][0]]
            CCF_list.append(CCF_norm)
            ACF_list.append(ACF_norm)
            ccf_dict[f'SNR_{molecule}']=SNR
            figs.CCF_plot(self,molecule,RVs,CCF_norm,ACF_norm,noiserange=noiserange)
        self.CCF_list=CCF_list
        self.ACF_list=ACF_list
        return ccf_dict

    def bayes_evidence(self,molecules,evidence_dict):

        bayes_dict=evidence_dict
        self.output_dir=pathlib.Path(f'{self.output_dir}/evidence_retrievals') # store output in separate folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(molecules, list)==False:
            molecules=[molecules] # if only one, make list so that it works in for loop

        for molecule in molecules: # exclude molecule from retrieval

            finish=pathlib.Path(f'{self.output_dir}/evidence_retrievals/final_wo{molecule}_posterior.npy')
            if finish.exists():
                print(f'\n ----------------- Evidence retrieval for {molecule} already done ----------------- \n')
                continue # check if already exists and continue if yes
            else:
                print(f'\n ----------------- Starting evidence retrieval for {molecule} ----------------- \n')

            if self.chemistry=='freechem':
                original_prior=self.parameters.param_priors[f'log_{molecule}']
                self.parameters.param_priors[f'log_{molecule}']=[-15,-14] # exclude from retrieval
            elif self.chemistry in ['equchem','quequchem']:
                if molecule=='13CO':
                    key='log_C12_13_ratio'
                elif molecule=='H2(18)O':
                    key='log_O16_18_ratio'
                original_prior=self.parameters.param_priors[key]
                self.parameters.param_priors[key]=[14,15] # exclude from retrieval

            self.callback_label=f'live_wo{molecule}_'
            self.prefix=f'pmn_wo{molecule}_' 
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance,resume=True)
            self.callback_label=f'final_wo{molecule}_'
            self.evaluate(callback_label=self.callback_label) # gets self.lnZ_ex

            ex_model=pRT_spectrum(self).make_spectrum()      
            lnL = self.LogLike(ex_model, self.Cov) # call function to generate chi2
            chi2_ex = self.LogLike.chi2_0_red # reduced chi^2

            print(f'lnZ=',self.lnZ)
            print(f'lnZ_{molecule}=',self.lnZ_ex)
            lnB,sigma=self.compare_evidence(self.lnZ, self.lnZ_ex)
            print(f'lnBm_{molecule}=',lnB)
            print(f'sigma_{molecule}=',sigma)
            print(f'chi2_{molecule}=',chi2_ex)
            bayes_dict[f'lnBm_{molecule}']=lnB
            bayes_dict[f'sigma_{molecule}']=sigma
            bayes_dict[f'chi2_wo_{molecule}']=chi2_ex
            print('\n ----------------- Current bayes_dict= ----------------- \n',bayes_dict)  

            # set back param priors for next retrieval
            if self.chemistry=='freechem':
                self.parameters.param_priors[f'log_{molecule}']=original_prior 
            elif self.chemistry in ['equchem','quequchem']:
                if molecule=='13CO':
                    key='log_C12_13_ratio'
                elif molecule=='H2(18)O':
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
        p = np.real(np.exp(W((-1.0/(np.exp(ln_B)*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)
        return ln_B*sign,sigma*sign

    def run_retrieval(self,N_live_points=400,evidence_tolerance=0.5,molecules=None,bayes=False): 
        self.N_live_points=N_live_points
        self.evidence_tolerance=evidence_tolerance
        retrieval_output_dir=self.output_dir # save end results here

        print(f'\n ---- {self.target.name} - chemistry: {self.chemistry} - PT type: {self.PT_type} - Nlive: {self.N_live_points} - ev: {self.evidence_tolerance} \n ----')

        # run main retrieval if hasn't been run yet, else skip to cross-corr and bayes
        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists()==False:
            print('\n ----------------- Starting main retrieval. ----------------- \n')
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance)
            save=True
        else:
            print('\n ----------------- Main retrieval exists. ----------------- \n')
            save=False
            with open(final_dict,'rb') as file:
                self.params_dict=pickle.load(file) 
        self.evaluate(save=save)
        if molecules!=None:
            ccf_dict=self.cross_correlation(molecules)
            self.params_dict.update(ccf_dict)
            with open(f'{retrieval_output_dir}/params_dict.pickle','wb') as file: # overwrite with added CCF SNR
                pickle.dump(self.params_dict,file)
        
        if bayes==True:
            evidence_dict=pathlib.Path(f'{retrieval_output_dir}/evidence_dict.pickle')
            if evidence_dict.exists()==False: # to avoid overwriting sigmas from other evidence retrievals
                print('\n ----------------- Creating evidence dict ----------------- \n')
                self.evidence_dict={}
            else:
                print('\n ----------------- Continuing exiting evidence dict ----------------- \n')
                with open(evidence_dict,'rb') as file:
                    self.evidence_dict=pickle.load(file)

            bayes_dict=self.bayes_evidence(molecules,evidence_dict=self.evidence_dict)
            print('\n ----------------- Final bayes_dict= ----------------- \n',bayes_dict)
            with open(f'{retrieval_output_dir}/evidence_dict.pickle','wb') as file: # save new results in separate dict
                pickle.dump(bayes_dict,file)

        output_file=pathlib.Path('retrieval.out')
        if output_file.exists():
            os.system(f"mv {output_file} {retrieval_output_dir}")

        print('\n ----------------- Done ---------------- \n')

        
        

