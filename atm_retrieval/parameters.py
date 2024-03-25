import numpy as np

class Parameters:

    params = {} # all parameters + their values

    def __init__(self, free_params, constant_params):
            
        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys())) # keys of free parameters
        self.n_params = len(self.param_keys) # number of free parameters
        self.ndim = self.n_params
        self.free_params=free_params
        self.constant_params=constant_params
        self.params.update(constant_params) # dictionary with constant parameter values
            
    @staticmethod
    def uniform_prior(bounds):
        return lambda x: x*(bounds[1]-bounds[0])+bounds[0]
    
    def __call__(self, cube, ndim=None, nparams=None):
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])
        
        for i, key_i in enumerate(self.param_keys):
            cube[i] = self.uniform_prior(self.param_priors[key_i])(cube[i]) # cube is vector of length nparams, values [0,1]
            self.params[key_i] = cube[i] # add free parameter values to parameter dictionary

        return self.cube_copy
