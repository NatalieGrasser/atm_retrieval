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
    from atm_retrieval.likelihood import Retrieval
    from atm_retrieval.parameters import Parameters
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    from target import Target
    from likelihood import Retrieval
    from parameters import Parameters

M0355 = Target('2M0355')
data_wave,data_flux,data_err=M0355.load_spectrum()
# replace atmosphere_objects file when adding new species!

constant_params = {'log_Kzz': 7.5, # eddy diffusion parameter (atmospheric mixing)
                'fsed': 2, # sedimentation parameter for particles
                'P_base_gray': 1, # pressure of gray cloud deck
                'fsed_gray': 2,
                'opa_base_gray': 0.8, # opacity of gray cloud deck
                'sigma_lnorm': 1.05, # width of the log-normal particle distribution of MgSiO3
                'log_MgSiO3' : 0, # scaling wrt chem equilibrium, 0 = equilibrium abundance 
                } 

# if free chemistry, define VMRs
# if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
free_params = {'rv': ([5,20],r'RV'),
               'vsini': ([0,40],r'RV'),
            'log_H2O':([-12,-1],r'H$_2$O'),
            'log_12CO':([-12,-1],r'$^{12}$CO'),
            'log_13CO':([-12,-1],r'$^{13}$CO'),
            'log_C18O':([-12,-1],r'C$^{18}$O'),
            'log_C17O':([-12,-1],r'C$^{17}$O'),
            'log_CH4':([-12,-1],r'CH$_4$'),
            'log_NH3':([-12,-1],r'NH$_3$'),
            'log_HCN':([-12,-1],r'HCN'),
            'T1' : ([2000,10000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
            'T2' : ([0,10000], r'$T_2$ [K]'),
            'T3' : ([0,10000], r'$T_3$ [K]'),
            'T4' : ([0,10000], r'$T_4$ [K]'), # top of atmosphere (cooler)
            'log_g':([2,5],r'log$g$')}

parameters = Parameters(free_params, constant_params)
cube = np.random.rand(parameters.n_params)
parameters(cube)
params=parameters.params

output='2M0355_test5'
retrieval=Retrieval(target=M0355,parameters=parameters,output_name=output)
retrieval.PMN_run(N_live_points=100,evidence_tolerance=5)
bestfit_model,final_params=retrieval.evaluate(plot_spectrum=True,plot_pt=True)