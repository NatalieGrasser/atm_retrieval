import numpy as np
from scipy.special import loggamma # gamma function

class LogLikelihood:

    def __init__(self,retr_obj,scale_flux=True,scale_err=True,alpha=2,N_phi=1):

        inherit_attributes = ['n_parts','n_pixels','primary_label','data_flux','mask_isfinite','data_wave']
        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(retr_obj, attr))

        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        self.N_d_total    = self.mask_isfinite.sum() # number of degrees of freedom / valid datapoints
        self.alpha = alpha # from Ruffio+2019
        self.N_phi = N_phi # number of linear scaling parameters

        if self.primary_label==False:
            self.mask_primary=np.empty((self.n_parts,self.n_pixels),dtype=bool)
            for i in range(self.n_parts):
                mask_i = np.isfinite(retr_obj.primary_flux[i]) # only finite pixels
                self.mask_primary[i]=mask_i
        
    def __call__(self, m_flux, Cov, **kwargs):

        self.ln_L   = 0.0
        self.chi2_0 = 0.0
        self.phi = np.ones((self.n_parts, self.N_phi)) # store linear flux-scaling terms
        self.s2  = np.ones((self.n_parts)) # uncertainty-scaling
        self.m_flux_phi = np.nan * np.ones_like(self.data_flux) # scaled model flux

        for i in range(self.n_parts): # Loop over all segments

            if self.primary_label==False:
                mask_i = self.mask_isfinite[i,:] & self.mask_primary[i,:]
            else:
                mask_i = self.mask_isfinite[i,:] # mask out nans
            N_d = mask_i.sum() # Number of (valid) data points in this order/det pair
            if N_d == 0:
                continue
            data_flux_i = self.data_flux[i,mask_i] # data flux
            m_flux_i = m_flux[i,mask_i] # model flux

            if not np.all(np.isfinite(m_flux_i)): # unresolved issue, quick fix for now
                model_mask_i = np.isfinite(m_flux_i)
                m_flux_i[~model_mask_i] = 2.0 # to distinguish from normalized values
                #import matplotlib.pyplot as plt
                #plt.plot(m_flux_i)
                #plt.savefig('model.png')
            
            if Cov[i].is_matrix:
                Cov[i].get_cholesky() # Retrieve a Cholesky decomposition
            if self.scale_flux: # Find the optimal phi-vector to match the observed spectrum
                self.m_flux_phi[i,mask_i],self.phi[i]=self.get_flux_scaling(data_flux_i, m_flux_i, Cov[i])

            residuals_phi = (self.data_flux[i] - self.m_flux_phi[i]) # Residuals wrt scaled model
            inv_cov_0_residuals_phi = Cov[i].solve(residuals_phi[mask_i])
            chi2_0 = np.dot(residuals_phi[mask_i].T, inv_cov_0_residuals_phi) # Chi-squared for the optimal linear scaling
            logdet_MT_inv_cov_0_M = 0

            if self.scale_flux:
                inv_cov_0_M    = Cov[i].solve(m_flux_i) # Covariance matrix of phi
                #print('inv_cov_0_M',inv_cov_0_M) # not nan
                MT_inv_cov_0_M = np.dot(m_flux_i.T, inv_cov_0_M)
                logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # (log)-determinant of the phi-covariance matrix

            if self.scale_err: 
                self.s2[i] = self.get_err_scaling(chi2_0, N_d) # Scale variance to maximize log-likelihood
            logdet_cov_0 = Cov[i].get_logdet()  # Get log of determinant (log prevents over/under-flow)

            # from Ruffio+2019
            self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi)+loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

            # Add this order/detector to the total log-likelihood
            self.ln_L += -1/2*(logdet_cov_0+logdet_MT_inv_cov_0_M+(N_d-self.N_phi+self.alpha-1)*np.log(chi2_0))
            self.chi2_0 += chi2_0/self.s2[i]
        
        self.chi2_red = self.chi2_0/self.N_d_total

        if np.isfinite(self.ln_L)==False:
            raise ValueError('Not finite lnL',self.ln_L,kwargs.get('params',None))
            return -np.inf
        else:
            return self.ln_L

    def get_flux_scaling(self, data_flux_i, m_flux_i, cov_i): 
        # Solve for linear scaling parameter phi: (M^T * cov^-1 * M) * phi = M^T * cov^-1 * d
        lhs = np.dot(m_flux_i.T, cov_i.solve(m_flux_i)) # Left-hand side
        rhs = np.dot(m_flux_i.T, cov_i.solve(data_flux_i)) # Right-hand side
        phi_i = rhs / lhs # Optimal linear scaling factor
        return np.dot(m_flux_i, phi_i), phi_i # Return scaled model flux + scaling factors

    def get_err_scaling(self, chi_squared_i_scaled, N_i):
        s2_i = np.sqrt(1/N_i * chi_squared_i_scaled)
        return s2_i # uncertainty scaling that maximizes log-likelihood