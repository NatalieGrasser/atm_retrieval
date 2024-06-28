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

class Retrieval:

    def __init__(self,target,parameters,output_name,scale_flux=True,free_chem=True,cloud_mode=None,lbl_opacity_sampling=3):

        self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])
        self.target=target
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.mask_isfinite=target.get_mask_isfinite() # shape (orders,detectors)
        self.separation,self.err_eff=target.prepare_for_covariance()
        self.parameters=parameters
        self.species=self.get_species(param_dict=self.parameters.params)

        self.n_orders, self.n_dets, _ = self.data_flux.shape # shape (orders,detectors,pixels)
        self.n_params = len(parameters.free_params)
        self.scale_flux= scale_flux
        self.free_chem=free_chem

        self.output_name=output_name
        self.output_dir = pathlib.Path(self.output_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # load atmosphere objects here and not in likelihood/pRT_model to make it faster
        self.do_scat_emis=False
        self.cloud_species=None
        if cloud_mode=='MgSiO3':
            self.cloud_species=['MgSiO3(c)_cd']
            self.do_scat_emis = True # enable scattering on cloud particles
        self.cloud_species=self.cloud_species
        self.lbl_opacity_sampling=lbl_opacity_sampling
        self.n_atm_layers=50
        self.pressure = np.logspace(-6,2,self.n_atm_layers)  # like in deRegt+2024

        self.Cov = np.empty((self.n_orders,self.n_dets), dtype=object) # covariance matrix
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                mask_ij = self.mask_isfinite[i,j] # only finite pixels
                if not mask_ij.any():
                    continue
                maxval=self.parameters.param_priors['l'][1]*3 # 3*max value of prior of l
                self.Cov[i,j] = CovGauss(err=self.data_err[i,j,mask_ij],separation=self.separation[i,j], 
                                         err_eff=self.err_eff[i,j],max_separation=maxval)
                
        self.LogLike = LogLikelihood(retrieval_object=self,scale_flux=True,scale_err=True)
        self.atmosphere_objects=self.get_atmosphere_objects()
        self.callback_label='live_' # label for plots
        self.bestfit_params=None # will be updated
        self.posterior = None
        self.final_params=None

    def get_species(self,param_dict): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        self.chem_species=[]
        for par in param_dict:
            if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
                if par in ['log_g','log_Kzz','log_MgSiO3','log_P_base_gray','log_opa_base_gray']: # skip those
                    pass
                else:
                    self.chem_species.append(par)
        species=[]
        for chemspec in self.chem_species:
            species.append(species_info.loc[chemspec[4:],'pRT_name'])
        return species

    def get_atmosphere_objects(self):

        atmosphere_objects=[]
        file=pathlib.Path(f'atmosphere_objects_{self.target.name}.pickle')
        if file.exists():
            with open(file,'rb') as file:
                atmosphere_objects=pickle.load(file)
                return atmosphere_objects
        else:
            for order in range(7):
                wl_pad=7 # wavelength padding because spectrum is not wavelength shifted yet
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
        self.model_flux=pRT_spectrum(parameters=self.parameters.params,data_wave=self.data_wave,target=self.target,
                                     atmosphere_objects=self.atmosphere_objects,species=self.species,
                                     free_chem=self.free_chem).make_spectrum()
        for j in range(self.n_orders): # update covariance matrix
            for k in range(self.n_dets):
                if not self.mask_isfinite[j,k].any():
                    continue
                self.Cov[j,k](self.parameters.params)
        ln_L = self.LogLike(self.model_flux, self.Cov) # retrieve log-likelihood
        return ln_L

    def PMN_run(self,N_live_points=200,evidence_tolerance=0.5,resume=False):
        pymultinest.run(LogLikelihood=self.PMN_lnL, Prior=self.parameters, 
                        n_dims=self.parameters.n_params, outputfiles_basename=f'{self.output_dir}/pmn_', verbose=True,
                        const_efficiency_mode=True, sampling_efficiency = 0.8,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # default is 0.5, high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=100)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        self.callback_label='live_' # label for plots
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        np.save(f'{self.output_dir}/{self.callback_label}bestfit_params.npy',self.bestfit_params)
        self.posterior = posterior[:,:-2] # remove last 2 columns
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.final_params,self.final_spectrum=self.get_final_params_and_spectrum()
        figs.make_all_plots(self)
     
    def PMN_analyse(self):
        self.callback_label='final_'
        analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                        outputfiles_basename=f'{self.output_dir}/pmn_')  # set up analyzer object
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
        self.posterior = self.posterior[:,:-1] # shape 
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # read params of best-fitting model, highest likelihood
        np.save(f'{self.output_dir}/{self.callback_label}bestfit_params.npy',self.bestfit_params)

    def get_quantiles(self,posterior,save=False):
        quantiles = np.array([np.percentile(posterior[:,i], [16.0,50.0,84.0], axis=-1) for i in range(posterior.shape[1])])
        medians=quantiles[:,1] # median of all params
        uppers=quantiles[:,2] # median+error
        lowers=quantiles[:,0] # median-error
        self.params_pm_dict={} # save in dictionary for easier access of all params
        for i,key in enumerate(self.parameters.param_keys):
            self.params_pm_dict[key]=[lowers[i],medians[i],uppers[i]]
        if save:
            np.save(f'{self.output_dir}/{self.callback_label}params_pm.npy',self.params_pm_dict)
        return medians,lowers,uppers

    def get_final_params_and_spectrum(self,contribution=True,use_median=True): 
        
        # make dict of constant params + evaluated params
        self.final_params=self.parameters.constant_params.copy() # initialize dict with constant params
        free_params_values=self.bestfit_params # use bestfit params with highest lnL (can differ from median, not as robust)
        if use_median==True:
            medians,_,_=self.get_quantiles(self.posterior,save=True)
            free_params_values=medians # use median of posterior as resulting final values, more robust
        for i,key in enumerate(self.parameters.param_keys):
            self.final_params[key]=free_params_values[i] # add evaluated params to constant params

        # create final spectrum to get phi_ij and s2_ij of bestfit model through likelihood
        self.final_object=pRT_spectrum(parameters=self.final_params,data_wave=self.data_wave,target=self.target,species=self.species,
                                        atmosphere_objects=self.atmosphere_objects,free_chem=self.free_chem,contribution=contribution)
        self.final_model=self.final_object.make_spectrum()
        self.log_likelihood = self.LogLike(self.final_model, self.Cov)
        self.final_params['phi_ij']=self.LogLike.phi
        self.final_params['s2_ij']=self.LogLike.s2
        np.save(f'{self.output_dir}/{self.callback_label}params_dict.npy',self.final_params)

        self.final_spectrum=np.zeros_like(self.final_model)
        phi_ij=self.final_params['phi_ij']
        for order in range(7):
            for det in range(3):
                self.final_spectrum[order,det]=phi_ij[order,det]*self.final_model[order,det] # scale model accordingly
        return self.final_params,self.final_spectrum

    def get_1213CO_FeH_CO(self): # can be run after self.evaluate()
        if self.final_params==None:
            self.final_params,_=self.get_final_params_and_spectrum()

        if self.free_chem==False:
            CO1213=1/self.final_params['C13_12_ratio']
            FeH=self.final_params['FEH']
            CO=self.params['C_O']
        if self.free_chem==True:
            VMR_12CO=10**(self.final_params['log_12CO'])
            VMR_13CO=10**(self.final_params['log_13CO'])
            CO1213=VMR_12CO/VMR_13CO
            FeH=self.final_object.FeH
            CO=self.final_object.CO
        return CO1213,FeH,CO # output 12CO/13CO, Fe/H, C/O

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True):
        self.callback_label='final_'
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.final_params,self.final_spectrum=self.get_final_params_and_spectrum() # all params: constant + free + scaling phi_ij + s2_ij
        figs.make_all_plots(self,only_abundances=only_abundances,only_params=only_params,split_corner=split_corner)

