import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import os
    os.environ['OMP_NUM_THREADS'] = '1' # important for MPI
    from mpi4py import MPI 
    comm = MPI.COMM_WORLD # important for MPI
    rank = comm.Get_rank() # important for MPI

import numpy as np
import pymultinest
from atm_retrieval.pRT_model import pRT_spectrum
from atm_retrieval.target import Target
from atm_retrieval.spectrum import Spectrum
import atm_retrieval.cloud_cond as cloud_cond
import corner
import pathlib
import matplotlib.pyplot as plt

class Retrieval:

    def __init__(self,target,parameters,output_name,scale_flux=True,free_chem=True):

        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.target=target

        self.parameters=parameters
        self.n_orders, self.n_dets, _ = self.data_flux.shape # shape (orders,detectors,pixels)
        self.n_params = len(parameters.free_params)
        self.scale_flux= scale_flux
        self.free_chem=free_chem

        self.output_name=output_name
        self.output_dir = pathlib.Path(self.output_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def likelihood(self,cube=None, ndim=None,nparams=None,plot=False):

        self.model_flux=pRT_spectrum(parameters=self.parameters.params,data_wave=self.data_wave,target=self.target,free_chem=self.free_chem).make_spectrum()
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
                    n_iter_before_update=30,n_live_points=N_live_points,resume=resume,
                    evidence_tolerance=evidence_tolerance) # default is 0.5, high number -> stops earlier

    def PMN_analyse(self):
        analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                        outputfiles_basename=f'{self.output_dir}/pmn_')  # Set-up analyzer object
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
        self.posterior = self.posterior[:,:-1] # shape      
        self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # Read the parameters of the best-fitting model

    def cornerplot(self):
        labels=list(self.parameters.free_params.keys())
        fig = corner.corner(self.posterior, 
                            labels=labels, 
                            title_kwargs={'fontsize': 12},
                            color='k',
                            linewidths=0.5,
                            fill_contours=True,
                            #quantiles=[0.16,0.84],
                            show_titles=True)
        if self.bestfit_params is not None:
            corner.overplot_lines(fig, self.bestfit_params,color='r',lw=1,linestyle='dashed')
        fig.savefig(f'{self.output_dir}/retrieval_summary.pdf')

    def get_final_parameters(self,contribution=False,get_spectrum=True): # make dict of constant params + determined bestfit params
        final_params=self.parameters.constant_params.copy()
        for i,key in enumerate(self.parameters.param_keys):
            final_params[key]=self.bestfit_params[i]
        if get_spectrum==True:
            self.final_object=pRT_spectrum(parameters=final_params,data_wave=self.data_wave,target=self.target,free_chem=self.free_chem,contribution=contribution)
            self.final_model=self.final_object.make_spectrum()
            self.log_likelihood = self.calculate_likelihood(self.data_flux,self.data_err,self.final_model) # to get f_ij and beta_ij of bestfit model
        final_params['f_ij']=self.f
        final_params['beta_ij']=self.beta
        self.final_params=final_params
        np.savetxt(f'{self.output_dir}/bestfit_params.txt')
        return final_params

    def get_bestfit_model(self,plot_spectrum=False,plot_pt=False):
        contribution=False
        if plot_pt==True:
            contribution=True
        f_ij=self.get_final_parameters(contribution=contribution)['f_ij']
        self.bestfit_flux=np.zeros_like(self.final_model)
        for order in range(7):
                for det in range(3):
                    self.bestfit_flux[order,det]=f_ij[order,det]*self.final_model[order,det] # scale model accordingly to get bestfit spectrum
        if plot_spectrum==True:
            fig,ax=plt.subplots(7,1,figsize=(9,9),dpi=200)
            for order in range(7):
                for det in range(3):
                    ax[order].plot(self.data_wave[order,det],self.data_flux[order,det],lw=0.8,alpha=0.8,c='k',label='data')
                    ax[order].plot(self.data_wave[order,det],self.bestfit_flux[order,det],lw=0.8,alpha=0.8,c='c',label='model')
                    #ax[order].yaxis.set_visible(False) # remove ylabels because anyway unitless
                    sigma=1
                    lower=self.data_flux[order,det]-self.data_err[order,det]*sigma
                    upper=self.data_flux[order,det]+self.data_err[order,det]*sigma
                    ax[order].fill_between(self.data_wave[order,det],lower,upper,color='k',alpha=0.2,label=f'{sigma}$\sigma$')
                    if order==0 and det==0:
                        ax[order].legend(fontsize=8) # to only have it once
                ax[order].set_xlim(np.nanmin(self.data_wave[order]),np.nanmax(self.data_wave[order]))
            ax[6].set_xlabel('Wavelength [nm]')
            fig.tight_layout(h_pad=-0.1)
            fig.savefig(f'{self.output_dir}/bestfit_spectrum.pdf')

        if plot_pt==True:
            summed_contr=np.mean(self.final_object.contr_em_orders,axis=0) # average over all orders
            if self.free_chem==False:
                C_O = self.final_object.params['C_O']
                Fe_H = self.final_object.params['FEH']
            if self.free_chem==True:
                C_O = self.final_object.CO
                Fe_H = self.final_object.FeH   
            fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=100)
            cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
            ax.plot(self.final_object.temperature, self.final_object.pressure)
            ax.scatter(self.final_object.t_samp,10**self.final_object.p_samp)
            ax.plot(summed_contr/np.max(summed_contr)*np.max(self.final_object.temperature),self.final_object.pressure,linestyle='dashed',lw=2)
            for cs in cloud_species:
                cs_key = cs[:-3]
                if cs_key == 'KCl':
                    cs_key = cs_key.upper()
                P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
                ax.plot(T_cloud, P_cloud, lw=2, label=cs, ls=':', alpha=0.8)
            ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]', yscale='log', 
                ylim=(np.nanmax(self.final_object.pressure),np.nanmin(self.final_object.pressure)),
                xlim=(400,np.nanmax(self.final_object.temperature)+200))
            ax.legend()
            fig.savefig(f'{self.output_dir}/PT_profile.pdf')

        return Spectrum(self.data_wave,self.bestfit_flux)
    
    def get_12CO_13CO_ratio(self):
        self.final_params=self.get_final_parameters(get_spectrum=False)
        if self.free_chem==False:
            return 1/self.final_params['C13_12_ratio']

        if self.free_chem==True:
            VMR_12CO=10**(self.final_params['log_12CO'])
            VMR_13CO=10**(self.final_params['log_13CO'])
            return VMR_12CO/VMR_13CO


