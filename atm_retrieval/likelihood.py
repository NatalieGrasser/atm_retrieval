import getpass
import os
if getpass.getuser() == "grasser": # when runnig from LEM
    os.environ['OMP_NUM_THREADS'] = '1' # important for MPI
    from mpi4py import MPI 
    comm = MPI.COMM_WORLD # important for MPI
    rank = comm.Get_rank() # important for MPI
    from atm_retrieval.pRT_model import pRT_spectrum
    import atm_retrieval.figures as figs
elif getpass.getuser() == "natalie": # when testing from my laptop
    from pRT_model import pRT_spectrum
    import figures as figs

import numpy as np
import pymultinest
import pathlib
import pickle
from petitRADTRANS import Radtrans
import pandas as pd

class Retrieval:

    def __init__(self,target,parameters,output_name,scale_flux=True,free_chem=True,cloud_mode=None,lbl_opacity_sampling=3):

        self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])

        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.target=target
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

        self.atmosphere_objects=self.get_atmosphere_objects()
        self.callback_label='live_' # label for plots

        # will be updated
        self.bestfit_params=None 
        self.posterior = None
        self.final_params=None

    def get_species(self,param_dict): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        self.chem_species=[]
        for par in param_dict:
            if 'log_' in par: # get all species in params dict
                if par in ['log_g','log_Kzz','log_MgSiO3','log_P_base_gray']: # skip those
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

    def calculate_likelihood(self,data,err,model): # deRegt+2024 eq.(3)

        self.ln_L = 0.0
        self.f = np.ones((self.n_orders, self.n_dets)) # store linear flux-scaling terms
        self.beta = np.ones((self.n_orders, self.n_dets)) # store uncertainty-scaling terms

        for i in range(self.n_orders): # Loop over all orders and detectors
            for j in range(self.n_dets):

                mask_ij = np.isfinite(data[i,j,:]) # mask out nans
                N_ij = mask_ij.sum() # Number of (valid) data points
                
                m_flux_ij = model[i,j,mask_ij] # model flux
                d_flux_ij = data[i,j,mask_ij] # data flux
                d_err_ij  = err[i,j,mask_ij] # data flux error
                res_ij = (d_flux_ij - m_flux_ij) # residuals

                cov_logdet = np.sum(np.log(d_err_ij**2)) # can be negative
                cov_ij = np.diag(d_err_ij**2) # covariance matrix
                inv_cov_ij = np.diag(1/d_err_ij**2) # inverse covariance matrix (if only diagonal)
                ln_L_ij = -0.5 * (N_ij*np.log(2*np.pi) + cov_logdet) # set up log-likelihood for this order/detector

                f_ij = 1.0
                if self.scale_flux: # Only scale the flux relative to the first order/detector
                    f_ij = self.get_flux_scaling(d_flux_ij, m_flux_ij, inv_cov_ij) # Scale the model flux to minimize the chi-squared error
                    m_flux_ij_scaled = f_ij*m_flux_ij
                    res_ij = (d_flux_ij - m_flux_ij_scaled)
                                
                chi_squared_ij_scaled = res_ij @ inv_cov_ij @ res_ij # chi-squared
                beta_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled) # optimal uncertainty scaling terms
                chi_squared_ij = 1/beta_ij**2 * chi_squared_ij_scaled # Chi-squared for optimal linear scaling and uncertainty scaling
                self.chi_squared_reduced = chi_squared_ij / (N_ij-self.n_params) # reduced chi-squared
                ln_L_ij += -(N_ij/2*np.log(beta_ij**2) + 1/2*chi_squared_ij) # add chi-squared and optimal uncertainty scaling terms to log-likelihood

                self.ln_L += ln_L_ij # add to total log-likelihood
                self.f[i,j] = f_ij # 'phi' in deRegt+2024 eq.(4)
                self.beta[i,j] = beta_ij # 's' in deRegt+2024 eq.(6)

        return self.ln_L
    
    def get_flux_scaling(self,d_flux_ij,m_flux_ij,inv_cov_ij): # deRegt+2024 eq.(5)
        lhs = m_flux_ij @ inv_cov_ij @ m_flux_ij # left-hand side
        rhs = m_flux_ij @ inv_cov_ij @ d_flux_ij # right-hand side
        f_ij = rhs / lhs
        return f_ij

    def likelihood(self,cube=None,ndim=None,nparams=None,plot=False):

        self.model_flux=pRT_spectrum(parameters=self.parameters.params,data_wave=self.data_wave,target=self.target,
                                     atmosphere_objects=self.atmosphere_objects,species=self.species,
                                     free_chem=self.free_chem).make_spectrum()
        #if plot: # just to check
            #fig = plt.figure(figsize=(10, 1),dpi=200)
            #plt.plot(self.data_wave.flatten(),self.data_flux.flatten())
            #plt.plot(self.data_wave.flatten(),self.model_flux.flatten())
            #plt.xlim(2422,2437)
        log_likelihood = self.calculate_likelihood(self.data_flux,self.data_err,self.model_flux)
        return log_likelihood 

    def PMN_run(self,N_live_points=200,evidence_tolerance=0.5,resume=False):
        result = pymultinest.run(LogLikelihood=self.likelihood, Prior=self.parameters,
                    n_dims=self.parameters.n_params, outputfiles_basename=f'{self.output_dir}/pmn_', verbose=True,
                    const_efficiency_mode=False, sampling_efficiency = 0.8,
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

        # create final spectrum to get f_ij and beta_ij of bestfit model through likelihood
        self.final_object=pRT_spectrum(parameters=self.final_params,data_wave=self.data_wave,target=self.target,species=self.species,
                                        atmosphere_objects=self.atmosphere_objects,free_chem=self.free_chem,contribution=contribution)
        self.final_model=self.final_object.make_spectrum()
        self.log_likelihood = self.calculate_likelihood(self.data_flux, self.data_err,self.final_model) 
        self.final_params['f_ij']=self.f
        self.final_params['beta_ij']=self.beta
        np.save(f'{self.output_dir}/{self.callback_label}params_dict.npy',self.final_params)

        self.final_spectrum=np.zeros_like(self.final_model)
        f_ij=self.final_params['f_ij']
        for order in range(7):
            for det in range(3):
                self.final_spectrum[order,det]=f_ij[order,det]*self.final_model[order,det] # scale model accordingly
        return self.final_params,self.final_spectrum

    def get_12CO_13CO_ratio(self):
        if self.final_params==None:
            self.final_params,_=self.get_final_params_and_spectrum()
        if self.free_chem==False:
            return 1/self.final_params['C13_12_ratio']
        if self.free_chem==True:
            VMR_12CO=10**(self.final_params['log_12CO'])
            VMR_13CO=10**(self.final_params['log_13CO'])
            return VMR_12CO/VMR_13CO

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True):
        self.callback_label='final_'
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.final_params,self.final_spectrum=self.get_final_params_and_spectrum() # all params: constant + free + scaling f_ij and beta_ij
        figs.make_all_plots(self,only_abundances=only_abundances,only_params=only_params,split_corner=split_corner)

