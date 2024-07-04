import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
    
class Covariance:
     
    def __init__(self, err, beta=None, **kwargs): 
        self.err = err
        self.cov_reset() # set up covariance matrix
        self.cov_cholesky = None # initialize
        self.beta=beta # uncertainty scaling factor 'beta' should be added to kwargs

    def __call__(self,params,**kwargs):
        self.cov_reset() # reset covariance matrix

    def cov_reset(self): # make diagonal covariance matrix from uncertainties
        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2) # = True if is matrix

    def add_data_err_scaling(self, beta): # Scale uncertainty with factor beta
        if not self.is_matrix:
            self.cov *= beta**2
        else:
            self.cov[np.diag_indices_from(self.cov)] *= beta**2

    def add_model_err(self, model_err): # Add a model uncertainty term
        if not self.is_matrix:
            self.cov += model_err**2
        else:
            self.cov += np.diag(model_err**2)

    def get_logdet(self): # log of determinant
        self.logdet = np.sum(np.log(self.cov)) 
        return self.logdet
 
    def solve(self, b): # Solve: cov*x = b, for x (x = cov^{-1}*b)
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal
    
    def get_dense_cov(self): 
        if self.is_matrix:
            return self.cov
        return np.diag(self.cov) # get the errors from the diagonal
    
class CovGauss: # covariance matrix suited for Gaussian processes

    def __init__(self, err, separation, err_eff=None, max_separation=None, **kwargs):
        
        # Pre-computed average error and wavelength separation
        self.err=err
        self.separation = np.abs(separation) # separation between pixels
        self.err_eff  = err_eff # average squared error between pixels
        self.cov_reset() # set up covariance matrix

        # Convert to banded matrices
        self.separation = self.get_banded(self.separation,max_value=max_separation,pad_value=1000) # pad high number bc will be truncated

    def get_banded(cls, array, max_value=None, pad_value=0, n_pixels=2048):
        banded_array = [] # Make banded covariance matrix
        for k in range(n_pixels):
            diag_k = np.diag(array, k=k) # Retrieve the k-th diagonal  
            diag_k = np.concatenate((diag_k, pad_value*np.ones(k))) # Pad the diagonals to the same sizes
            if (diag_k == 0).all() and (k != 0): # There are no more non-zero diagonals coming
                break
            if max_value is not None:
                if (diag_k > max_value).all():
                    break
            banded_array.append(diag_k)
        return np.asarray(banded_array) # Convert to array for scipy
    
    def __call__(self,params,**kwargs):
        self.cov_reset() # Reset covariance matrix
        a = 10**(params.get('log_a'))
        l = 10**(params.get('log_l'))
        if (a is not None) and (l is not None): 
            self.add_RBF_kernel(a=a,l=l,variance=self.err_eff, **kwargs) # add radial-basis function kernel
            
    def cov_reset(self): # Create covariance matrix from uncertainties
        self.cov = np.zeros_like(self.separation)
        self.cov[0] = self.err**2
        self.is_matrix = True

    def add_RBF_kernel(self, a, l, variance, trunc_dist=5, scale_GP_amp=True, **kwargs):
        # a=square-root of amplitude of RBF kernel
        # l=length-scale of RBF kernel
        # trunc_dist=where to truncate kernel
        w_ij = (self.separation < trunc_dist*l) # Hann window function to ensure sparsity
        GP_amp = a**2 # GP amplitude
        if scale_GP_amp: # Use amplitude as fraction of flux uncertainty
            if isinstance(variance, float):
                GP_amp *= variance**2
            else:
                GP_amp *= variance[w_ij]**2
        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2)) # Gaussian radial-basis function kernel

    def get_cholesky(self):
        self.cov = self.cov[(self.cov!=0).any(axis=1),:] # input banded cov (is banded bc separation is)
        self.cov_cholesky = cholesky_banded(self.cov, lower=True) # banded Cholesky decomposition w scipy 

    def get_logdet(self): # log of determinant of banded Cholesky
        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))
        return self.logdet
    
    def solve(self, b): # solve cov*x = b, for x (x = cov^{-1}*b) for banded Cholesky
        return cho_solve_banded((self.cov_cholesky, True), b)
    
    def get_dense_cov(self):
        cov_full = np.zeros((self.cov.shape[1], self.cov.shape[1])) # Full covariance matrix
        for i, diag_i in enumerate(self.cov):
            if i != 0:
                diag_i = diag_i[:-i]
            cov_full += np.diag(diag_i, k=i) # Fill upper diagonals
            if i != 0:
                cov_full += np.diag(diag_i, k=-i) # Fill lower diagonals
        return cov_full





