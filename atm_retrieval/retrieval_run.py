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

testing=True

if testing:

    constant_params = {'vsini': 2, # rotational velocity
                    'rv': 11.92,
                    'log_Kzz': 7.5, # eddy diffusion parameter (atmospheric mixing)
                    'fsed': 2, # sedimentation parameter for particles
                    'P_base_gray': 1, # pressure of gray cloud deck
                    'fsed_gray': 2,
                    'opa_base_gray': 0.8, # opacity of gray cloud deck
                    'sigma_lnorm': 1.05, # width of the log-normal particle distribution of MgSiO3
                    'log_MgSiO3' : 0, # scaling wrt chem equilibrium, 0 = equilibrium abundance 
                    } 

    # if free chemistry, define VMRs
    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    free_params = {'log_H2O':([-12,-1],r'H$_2$O'),
                'log_12CO':([-12,-1],r'$^{12}$CO'),
                'log_13CO':([-12,-1],r'$^{13}$CO'),
                'log_C18O':([-12,-1],r'C$^{18}$O'),
                'log_C17O':([-12,-1],r'C$^{17}$O'),
                'log_CH4':([-12,-1],r'CH$_4$'),
                'log_NH3':([-12,-1],r'NH$_3$'),
                'log_HCN':([-12,-1],r'HCN'),
                'T1' : ([1000,5000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
                'T2' : ([700,3000], r'$T_2$ [K]'),
                'T3' : ([300,2000], r'$T_3$ [K]'),
                'T4' : ([200,1000], r'$T_4$ [K]'), # top of atmosphere (cooler)
                'log_g':([2,5],r'log$g$'),
                }
    
else:
    constant_params = {#'log_g': 4.65,
                        #'T1': 2500,
                        #'T2': 1800,
                        #'T3': 1400,
                        #'T4': 1100,
                        #'T5': 900,
                        #'T6': 800,
                        #'T7': 700,
                        #'log_H2O':-3,
                        #'log_12CO':-3,
                        #'log_13CO':-np.inf,
                        #'log_C18O':-np.inf,
                        #'log_C17O':-np.inf,
                        #'log_CH4':-np.inf,
                        #'log_NH3':-np.inf,
                        #'log_HCN':-np.inf,
                        #'vsini': 2, # rotational velocity
                        #'rv': 11.92,
                        'log_Kzz': 7.5, # eddy diffusion parameter (atmospheric mixing)
                        'fsed': 2, # sedimentation parameter for particles
                        'P_base_gray': 1, # pressure of gray cloud deck
                        'fsed_gray': 2,
                        'opa_base_gray': 0.8, # opacity of gray cloud deck
                        'sigma_lnorm': 1.05, # width of the log-normal particle distribution of MgSiO3
                        'log_MgSiO3' : 0, # scaling wrt chem equilibrium, 0 = equilibrium abundance 
                        } 

    # if free chemistry, define VMRs
    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    free_params = {'vsini':([1.0,10.0],r'$v \sin(i)$ [km/s]'),
                'rv':([5.0,20.0],r'RV [km/s]'),
                'log_H2O':([-5,-2],r'H$_2$O'),
                'log_12CO':([-5,-2],r'$^{12}$CO'),
                'log_13CO':([-12,-1],r'$^{13}$CO'),
                'log_C18O':([-12,-1],r'C$^{18}$O'),
                'log_C17O':([-12,-1],r'C$^{17}$O'),
                'log_CH4':([-12,-1],r'CH$_4$'),
                'log_NH3':([-12,-1],r'NH$_3$'),
                'log_HCN':([-12,-1],r'HCN'),
                #'C_O':([0,1],r'C/O'),
                #'FEH':([-2,2],r'Fe/H'),
                #'C13_12_ratio':([0,1],r'13C/12C'),
                #'O18_16_ratio':([0,1],r'18O/16O'),
                #'O17_16_ratio':([0,1],r'17O/16O'),
                'T1' : ([1500, 7000], r'$T_1$ [K]'), # bottom of the atmosphere (hotter)
                'T2' : ([1000, 5000], r'$T_2$ [K]'),
                'T3' : ([700, 3000], r'$T_3$ [K]'),
                'T4' : ([500,  2000], r'$T_4$ [K]'),
                'T5' : ([400,  1500], r'$T_5$ [K]'),
                'T6' : ([300,  1300],  r'$T_6$ [K]'),
                'T7' : ([200,  1000],  r'$T_7$ [K]'), # top of atmosphere (cooler)
                'log_g':([2,5],r'log$g$'),
                }

parameters = Parameters(free_params, constant_params)
cube = np.random.rand(parameters.n_params)
parameters(cube)
params=parameters.params

output='2M0355_test4'
retrieval=Retrieval(target=M0355,parameters=parameters,output_name=output)
retrieval.PMN_run(N_live_points=100,evidence_tolerance=5)
bestfit_model,final_params=retrieval.evaluate(plot_spectrum=True,plot_pt=True)