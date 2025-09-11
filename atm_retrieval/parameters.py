import numpy as np
import re

class Parameters:

    def __init__(self, free_params, constant_params):

        self.params = {} # all parameters + their values
            
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

        # get all T knot keys, start with "T" followed by digits, exclude T0
        self.t_keys = [key for key in self.param_keys if re.fullmatch(r"T\d+", key) and key!='T0'] 
        self.t_keys = sorted(self.t_keys, key=lambda x: int(x[1:]))

        self.t_grad_keys = [key for key in self.param_keys if re.fullmatch(r"dlnT_dlnP_\d+", key)] 

        # for varchem
        self.var_keys=[]
        if any('log_H2O_' in key for key in self.param_keys):
            n_knots = sum(1 for key in self.param_keys if re.fullmatch(r'log_H2O_\d+', key))
            self.var_keys = [f'log_H2O_{int(i)}' for i in range(n_knots)][1:]
        if any('log_p_H2O_' in key for key in self.param_keys):
            n_knots = sum(1 for key in self.param_keys if re.fullmatch(r'log_p_H2O_\d+', key))
            self.var_keys = [f'log_p_H2O_{int(i)}' for i in range(n_knots)][1:]

        # for varying pressure points
        self.p_RCB=''
        if 'log_P_RCB' in self.param_keys:
            self.p_RCB = 'log_P_RCB'
            n_array = np.linspace(1,len(t_grad_keys),len(t_grad_keys),dtype=int)[:-2]
            n_RCB = int(np.median(n_array))
            self.tgrad_RCB = t_grad_keys[n_RCB]
            self.tgrad_before = t_grad_keys[:n_RCB]
            self.tgrad_after = t_grad_keys[n_RCB+1:]

        # for partial pressures
        self.log_p_keys = [key for key in self.param_keys if 'log_p_' in key]
        self.p_sum = 0.0
        self.p_max = 10**self.constant_params['log_P_upper']
            
    @staticmethod
    def uniform_prior(bounds):
        return lambda x: x*(bounds[1]-bounds[0])+bounds[0]
    
    def __call__(self, cube, ndim=None, nparams=None):
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])

        for i, key_i in enumerate(self.param_keys):
            
            if key_i not in self.t_keys + self.var_keys + self.log_p_keys:  # to not set cube[i] for T1-T4 beforehand, must stay [0,1]
                cube[i] = self.uniform_prior(self.param_priors[key_i])(cube[i]) # cube is vector of length nparams, values [0,1]
            
            # allow only minor temperature inversions
            if key_i in self.t_keys: # as long as order in dict T0,...
                cube[i]=self.uniform_prior([cube[i-1]*0.5,cube[i-1]*1.05])(cube[i]) # like in Zhang+2021 on 2M0355

            if key_i in self.t_grad_keys and self.p_RCB!='': # PTgradvar, make sure largest grad at RCB
                #print(key_i)
                if key_i in self.tgrad_before: # do as usual
                    cube[i] = self.uniform_prior(self.param_priors[key_i])(cube[i])
                    #print('before',cube[i])
                elif key_i== self.tgrad_RCB: # must be larger than the ones 
                    #('cube[i-len(tgrad_before):i]',cube[i-len(tgrad_before):i])
                    maxgrad = max(cube[i-len(self.tgrad_before):i])#max(cube[i-1],cube[i-2])# values of previous tgrad
                    cube[i] = self.uniform_prior([maxgrad,self.param_priors[key_i][-1]])(cube[i])
                    #print('RCB',maxgrad,cube[i])
                elif key_i in self.tgrad_after: # rest should be smaller than tgrad at RCB
                    #print('cube[i-(len(tgrad_before)+1):i]',cube[i-(len(tgrad_before)+1):i])
                    maxgrad = max(cube[i-(len(self.tgrad_before)+1):i]) #max(cube[i-1],cube[i-2],cube[i-3])
                    cube[i] = self.uniform_prior([self.param_priors[key_i][0],maxgrad])(cube[i])
                    #print('after',maxgrad,cube[i])
                #print(key_i,cube[i])
            
            #if key_i in p_keys: # start at bottom of atmosphere, pressure must decrease
                #if key_i=='log_P_1': # set first free param (log_P_0 is const, at bottom of atm)
                    #cube[i]=self.uniform_prior(self.param_priors[key_i])(cube[i])
                #else:
                    #cube[i]=min(cube[i],cube[i-1])*(1+1e-1)
                
            # no temperature inversion for isolated objects, so force temperature to increase to avoid weird fluctuations
            #if key_i in ["T2","T3","T4","T5"]: # take value equal to or smaller than previous
                #cube[i]=min(cube[i],cube[i-1]) # as long as order in dict T1,T2,T3,T4

            if key_i in self.var_keys: # abundance decrease to top, allow minor increase only
                cube[i]=self.uniform_prior([cube[i-1]*1.1,self.param_priors[key_i][-1]])(cube[i])

            if key_i in self.log_p_keys:
                cube[i] = self.uniform_prior(self.param_priors[key_i])(cube[i])
                self.p_sum += 10**cube[i]
                #print(key_i,cube[i],p_sum)
                if self.p_sum >= self.p_max:
                    #print('ERROR',key_i,cube[i],p_sum)
                    #cube[i-1] = self.param_priors[key_i][0]
                    cube[i] = self.param_priors[key_i][0] # lowest value

            self.params[key_i] = cube[i] # add free parameter values to parameter dictionary

        return self.cube_copy
