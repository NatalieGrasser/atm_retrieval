import getpass
import os
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs

if getpass.getuser() == "grasser": # when runnig from LEM
    os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"
    os.environ['OMP_NUM_THREADS'] = '1' # important for MPI
    from mpi4py import MPI 
    comm = MPI.COMM_WORLD # important for MPI
    rank = comm.Get_rank() # important for MPI
    from atm_retrieval.target import Target
    from atm_retrieval.retrieval import Retrieval
    from atm_retrieval.parameters import Parameters
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    from target import Target
    from retrieval import Retrieval
    from parameters import Parameters

M0355 = Target('2M0355')
data_wave,data_flux,data_err=M0355.load_spectrum()
# replace atmosphere_objects file when adding new species!

constant_params = {
                #'fsed': 2, # sedimentation parameter for particles
                #'sigma_lnorm': 1.05, # width of the log-normal particle distribution of MgSiO3
                #'log_MgSiO3' : 0, # scaling wrt chem equilibrium, 0 = equilibrium abundance 
                } 

# if free chemistry, define VMRs
# if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
free_params = {'rv': ([5,20],r'$v_{\rm rad}$'),
               'vsini': ([0,30],r'$v$ sin$i$'),

            # Cloud properties
            'log_opa_base_gray': ([-10,3], r'log $\kappa_{\mathrm{cl},0}$'), 
            'log_P_base_gray': ([-6,3], r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
            'fsed_gray': ([0,20], r'$f_\mathrm{sed}$'), # sedimentation parameter for particles

            # Chemistry
            'log_H2O':([-12,-1],r'log H$_2$O'),
            'log_12CO':([-12,-1],r'log $^{12}$CO'),
            'log_13CO':([-12,-1],r'log $^{13}$CO'),
            'log_C18O':([-12,-1],r'log C$^{18}$O'),
            'log_C17O':([-12,-1],r'log C$^{17}$O'),
            'log_CH4':([-12,-1],r'log CH$_4$'),
            'log_NH3':([-12,-1],r'log NH$_3$'),
            'log_HCN':([-12,-1],r'log HCN'),
            'log_HF':([-12,-1],r'log HF'),
            'log_H2(18)O':([-12,-1],r'log H$_2^{18}$O'),
            'log_H2S':([-12,-1],r'log H$_2$S'),
            'log_OH':([-12,-1],r'log OH'),
            'log_CO2':([-12,-1],r'log CO$_2$'),

            # P-T profile
            'T1' : ([1000,4000], r'$T_1$'), # bottom of the atmosphere (hotter)
            'T2' : ([0,4000], r'$T_2$'),
            'T3' : ([0,4000], r'$T_3$'),
            'T4' : ([0,4000], r'$T_4$'), # top of atmosphere (cooler)

            # others
            'log_g':([3,5],r'log $g$'),
            'log_Kzz':([5,15],r'log $K_{zz}$'), # eddy diffusion parameter (atmospheric mixing)
            'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], # limb-darkening coefficient (0-1) 

            # Uncertainty scaling
            'a': [(0.1,0.8), r'$a$'], # one is enough, will be multipled with order/det error
            'l': [(10,40), r'$l$']
            }

parameters = Parameters(free_params, constant_params)
cube = np.random.rand(parameters.n_params)
parameters(cube)
params=parameters.params

output='2M0355_test12'
retrieval=Retrieval(target=M0355,parameters=parameters,output_name=output,cloud_mode='gray')
retrieval.PMN_run(N_live_points=100,evidence_tolerance=5)
#retrieval.PMN_run(N_live_points=200,evidence_tolerance=0.5)
#only_params=['vsini','log_H2O','log_12CO','log_13CO','T1','T2','T3','T4']
retrieval.evaluate()
