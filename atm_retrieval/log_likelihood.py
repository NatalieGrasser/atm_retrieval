import numpy as np
from scipy.special import loggamma # gamma function

class LogLikelihood:

    def __init__(self,retrieval_object,scale_flux=False,scale_err=False,alpha=2,N_phi=1):

        self.d_flux = retrieval_object.data_flux
        self.d_mask = retrieval_object.mask_isfinite
        self.n_orders = retrieval_object.n_orders
        self.n_dets   = retrieval_object.n_dets
        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        self.N_d      = self.d_mask.sum() # number of degrees of freedom / valid datapoints
        self.N_params = retrieval_object.n_params
        self.alpha = alpha # from Ruffio+2019
        self.N_phi = N_phi # number of linear scaling parameters
        
    def __call__(self, m_flux, Cov, **kwargs):

        self.ln_L   = 0.0
        self.chi2_0 = 0.0
        self.phi = np.ones((self.n_orders, self.n_dets, self.N_phi)) # store linear flux-scaling terms
        self.s2  = np.ones((self.n_orders, self.n_dets)) # uncertainty-scaling
        self.m_flux_phi = np.nan * np.ones_like(self.d_flux) # scaled model flux

        for i in range(self.n_orders): # Loop over all orders and detectors
            for j in range(self.n_dets):

                mask_ij = self.d_mask[i,j,:] # mask out nans
                N_d = mask_ij.sum() # Number of (valid) data points
                if N_d == 0:
                    continue
                d_flux_ij = self.d_flux[i,j,mask_ij] # data flux
                m_flux_ij = m_flux[i,j,mask_ij] # model flux
                
                if Cov[i,j].is_matrix:
                    Cov[i,j].get_cholesky() # Retrieve a Cholesky decomposition
                if self.scale_flux: # Find the optimal phi-vector to match the observed spectrum
                    self.m_flux_phi[i,j,mask_ij],self.phi[i,j]=self.get_flux_scaling(d_flux_ij, m_flux_ij, Cov[i,j])

                residuals_phi = (self.d_flux[i,j] - self.m_flux_phi[i,j]) # Residuals wrt scaled model
                inv_cov_0_residuals_phi = Cov[i,j].solve(residuals_phi[mask_ij]) 
                chi2_0 = np.dot(residuals_phi[mask_ij].T, inv_cov_0_residuals_phi) # Chi-squared for the optimal linear scaling
                logdet_MT_inv_cov_0_M = 0

                if self.scale_flux:
                    inv_cov_0_M    = Cov[i,j].solve(m_flux_ij) # Covariance matrix of phi
                    MT_inv_cov_0_M = np.dot(m_flux_ij.T, inv_cov_0_M)
                    logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # (log)-determinant of the phi-covariance matrix

                if self.scale_err: 
                    self.s2[i,j] = self.get_err_scaling(chi2_0, N_d) # Scale variance to maximize log-likelihood
                logdet_cov_0 = Cov[i,j].get_logdet()  # Get log of determinant (log prevents over/under-flow)

                # not same as in deRegt+2024, but taken from Ruffio+2019, Sam did not inlcude it
                self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi)+loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

                # Add this order/detector to the total log-likelihood
                self.ln_L += -1/2*(logdet_cov_0+logdet_MT_inv_cov_0_M+(N_d-self.N_phi+self.alpha-1)*np.log(chi2_0))
                self.chi2_0 += chi2_0

        # Reduced chi-squared
        self.chi2_0_red = self.chi2_0 / self.N_d

        return self.ln_L

    def get_flux_scaling(self, d_flux_ij, m_flux_ij, cov_ij): 
        # Solve for linear scaling parameter phi: (M^T * cov^-1 * M) * phi = M^T * cov^-1 * d
        lhs = np.dot(m_flux_ij.T, cov_ij.solve(m_flux_ij)) # Left-hand side
        rhs = np.dot(m_flux_ij.T, cov_ij.solve(d_flux_ij)) # Right-hand side
        phi_ij = rhs / lhs # Optimal linear scaling factor
        return np.dot(m_flux_ij, phi_ij), phi_ij # Return scaled model flux + scaling factors

    def get_err_scaling(self, chi_squared_ij_scaled, N_ij):
        s2_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled)
        return s2_ij # uncertainty scaling that maximizes log-likelihood