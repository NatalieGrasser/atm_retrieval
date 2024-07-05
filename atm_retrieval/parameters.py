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
            
            if key_i not in ["T1","T2","T3","T4"]:  # to not set cube[i] for T1-T4 beforehand, must stay [0,1]
                cube[i] = self.uniform_prior(self.param_priors[key_i])(cube[i]) # cube is vector of length nparams, values [0,1]
            
            if key_i in ["T1","T2","T3","T4"]: # as long as order in dict T0,T1,T2,T3,T4
                cube[i]=self.uniform_prior([cube[i-1]*0.5,cube[i-1]])(cube[i]) # like in Zhang+2021 on 2M0355
                
            # no temperature inversion for isolated objects, so force temperature to increase to avoid weird fluctuations
            #if key_i in ["T2","T3","T4","T5"]: # take value equal to or smaller than previous
                #cube[i]=min(cube[i],cube[i-1]) # as long as order in dict T1,T2,T3,T4
            self.params[key_i] = cube[i] # add free parameter values to parameter dictionary

        return self.cube_copy
