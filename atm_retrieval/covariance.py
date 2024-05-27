import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

def get_Covariance_class(err, **kwargs): 

    if kwargs.get('cov_mode') == 'GP': # Use a GaussianProcesses instance
        return GaussianProcesses(err, **kwargs)
    else:
        return Covariance(err, **kwargs) # Use a Covariance instance instead
    
class Covariance:
     
    def __init__(self, err, beta=None, **kwargs): 

        self.err = err
        self.cov_reset() # set up covariance matrix
        self.cov_cholesky = None # initialize
        self.beta=beta # uncertainty scaling factor 'beta' should be added to kwargs

    def __call__(self, params, w_set, order, det, **kwargs):

        self.cov_reset() # Reset the covariance matrix  
        if params.get(f'beta_{w_set}') is None: # uncertainty scaling factor for wavelength set (order,det)
            return
        
        if params[f'beta_{w_set}'][order,det] != 1:
            self.add_data_err_scaling(params[f'beta_{w_set}'][order,det])

    def cov_reset(self): # Create covariance matrix from uncertainties

        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2) # = True if is matrix
        self.cov_shape = self.cov.shape

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
 
    def solve(self, b): # Solve: cov*x = b, for x (x = cov^{-1}*b)
        
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal
    
    def get_dense_cov(self): 

        if self.is_matrix:
            return self.cov
        return np.diag(self.cov) # get the errors from the diagonal
    

class GaussianProcesses: # covariance matrix suited for Gaussian processes

    def __init__(self, err, separation, err_eff=None, flux_eff=None, max_separation=None, **kwargs):
        # separation between pixels
        #err_eff : average squared error between pixels

        # Pre-computed average error and wavelength separation
        self.separation = np.abs(separation)
        self.err_eff  = err_eff
        self.flux_eff = flux_eff

        # Convert to banded matrices
        self.separation = self.get_banded(self.separation, max_value=max_separation, pad_value=1000)

    def get_banded(cls, array, max_value=None, pad_value=0, n_pixels=2048):

        banded_array = [] # Make banded covariance matrix
        for k in range(n_pixels):
            if array.dtype == object: # Array consists of order-detector pairs
                n_orders, n_dets = array.shape
                diag_k = []
                for i in range(n_orders):
                    for j in range(n_dets):
                        diag_ijk = np.diag(array[i,j], k=k) # Retrieve the k-th diagonal
                        diag_ijk = np.concatenate((diag_ijk, pad_value*np.ones(k)))  # Pad to the same sizes
                        diag_k.append(diag_ijk) # Append to diagonals of other order/detectors
                diag_k = np.concatenate(diag_k)
            else:
                diag_k = np.diag(array, k=k) # Retrieve the k-th diagonal   
                diag_k = np.concatenate((diag_k, pad_value*np.ones(k))) # Pad the diagonals to the same sizes

            if (diag_k == 0).all() and (k != 0): # There are no more non-zero diagonals coming
                break
            if max_value is not None:
                if (diag_k > max_value).all():
                    break

            banded_array.append(diag_k)

        return np.asarray(banded_array) # Convert to array for scipy
    
    def __call__(self, params, w_set, order=0, det=0, **kwargs):

        self.cov_reset() # Reset the covariance matrix
        beta = params.get('beta', params.get(f'beta_{w_set}'))
        a = params.get('a', params.get(f'a_{w_set}'))
        l = params.get('l', params.get(f'l_{w_set}'))

        if beta is not None:
            self.add_data_err_scaling(params[f'beta_{w_set}'][order,det])
        
        if (a is not None) and (l is not None): # Add a radial-basis function kernel
            self.add_RBF_kernel(a=a[order,det], 
                                l=l[order,det], 
                                array=self.err_eff, **kwargs)
            
    def cov_reset(self): # Create the covariance matrix from the uncertainties
        self.cov = np.zeros_like(self.separation)
        self.cov[0] = self.err**2
        self.is_matrix = True

    def add_RBF_kernel(self, a, l, array, trunc_dist=5, scale_GP_amp=False, **kwargs):
        '''
        Add a radial-basis function kernel to the covariance matrix. 
        The amplitude can be scaled by the flux-uncertainties of 
        pixels i and j if scale_GP_amp=True. 

        Input
        -----
        a : float
            Square-root of amplitude of the RBF kernel.
        l : float
            Length-scale of the RBF kernel.
        trunc_dist : float
            Distance at which to truncate the kernel 
            (|wave_i-wave_j| < trunc_dist*l). This ensures
            a relatively sparse covariance matrix. 
        scale_GP_amp : bool
            If True, scale the amplitude at each covariance element, 
            using the flux-uncertainties of the corresponding pixels
            (A = a**2 * (err_i**2 + err_j**2)/2).
        '''

    
    # WHAT IS W_IJ?

        w_ij = (self.separation < trunc_dist*l) # Hann window function to ensure sparsity
        GP_amp = a**2 # GP amplitude
        if scale_GP_amp: # Use amplitude as fraction of flux uncertainty
            if isinstance(array, float):
                GP_amp *= array**2
            else:
                GP_amp *= array[w_ij]**2

        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2)) # Gaussian radial-basis function kernel


    def get_cholesky(self):
        self.cov = self.cov[(self.cov!=0).any(axis=1),:]
        self.cov_cholesky = cholesky_banded(self.cov, lower=True) # banded Cholesky decomposition w scipy function 

    def get_logdet(self): # log of the determinant of banded Cholesky
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





