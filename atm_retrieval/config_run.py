import getpass
import os
import numpy as np
import sys
os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs

if getpass.getuser() == "grasser": # when running from LEM
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

# default configuration
brown_dwarf = '2M0355' # options: 2M0355 or 2M1425 or test
chem = 'freechem' # options: freechem or equchem
PT_type = 'PTgrad' # options: PTknot or PTgrad
Nlive=400 # number of live points
tol=0.5 # evidence tolerance

# pass configuration as command line argument
# example: config_run.py 2M0355 freechem PTgrad
if len(sys.argv)>1:
    brown_dwarf = sys.argv[1] # options: 2M0355 or 2M1425 or test
    chem = sys.argv[2] # options: freechem or equchem
    PT_type = sys.argv[3] # options: PTknot or PTgrad
output=f'{brown_dwarf}_{chem}_{PT_type}' # output folder name

# option to change live points and evidence tolerance
# example: config_run.py 2M0355 freechem PTgrad 200 5
if len(sys.argv)>4:
    Nlive=int(sys.argv[4])
    tol=float(sys.argv[5])
    output=f'{chem}_{PT_type}_N{Nlive}_ev{tol}' # output folder name
    
brown_dwarf = Target(brown_dwarf)
cloud_mode='gray' # options: None, gray, or MgSiO3
GP=True # options: True/False

constant_params={} # add if needed
free_params = {'rv': ([2,20],r'$v_{\rm rad}$'),
               'vsini': ([0,40],r'$v$ sin$i$'),
               'log_g':([3,5],r'log $g$'),
               'epsilon_limb': ([0.2,1], r'$\epsilon_\mathrm{limb}$')} # limb-darkening coefficient (0-1)

if PT_type=='PTknot':
    pt_params={'T0' : ([1000,4000], r'$T_0$'), # bottom of the atmosphere (hotter)
            'T1' : ([0,4000], r'$T_1$'),
            'T2' : ([0,4000], r'$T_2$'),
            'T3' : ([0,4000], r'$T_3$'),
            'T4' : ([0,4000], r'$T_4$'),} # top of atmosphere (cooler)
    free_params.update(pt_params)

if PT_type=='PTgrad':
    pt_params={'dlnT_dlnP_0': ([0.,0.4], r'$\nabla T_0$'), # gradient at T0 
            'dlnT_dlnP_1': ([0.,0.4], r'$\nabla T_1$'), 
            'dlnT_dlnP_2': ([0.,0.4], r'$\nabla T_2$'), 
            'dlnT_dlnP_3': ([0.,0.4], r'$\nabla T_3$'), 
            'dlnT_dlnP_4': ([0.,0.4], r'$\nabla T_4$'), 
            'T0': ([1000,4000], r'$T_0$')} # at bottom of atmosphere
    free_params.update(pt_params)

# if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
if chem=='equchem':
    chemistry={'C/O':([0,1], r'C/O'), 
            'Fe/H': ([-1.5,1.5], r'[Fe/H]'), 
            'C13_12_ratio': ([1e-10,1e-1], r'$\mathrm{^{13}C/^{12}C}$'), 
            'O18_16_ratio': ([1e-10,1e-1], r'$\mathrm{^{18}O/^{16}O}$'), 
            'O17_16_ratio': ([1e-10,1e-1], r'$\mathrm{^{17}O/^{12}O}$')}
    
# if free chemistry, define VMRs
if chem=='freechem': 
    chemistry={'log_H2O':([-12,-1],r'log H$_2$O'),
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
            #'log_OH':([-12,-1],r'log OH'),
            #'log_CO2':([-12,-1],r'log CO$_2$')
            }
    
if cloud_mode=='gray':
    cloud_props={'log_opa_base_gray': ([-10,3], r'log $\kappa_{\mathrm{cl},0}$'),  
                'log_P_base_gray': ([-6,3], r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
                'fsed_gray': ([0,20], r'$f_\mathrm{sed}$')} # sedimentation parameter for particles
    free_params.update(cloud_props)

if cloud_mode=='MgSiO3':
    cloud_props={'fsed': ([0,20], r'$f_\mathrm{sed}$'), # sedimentation parameter for particles
                'sigma_lnorm': ([0.8,1.5], r'$\sigma_{l,norm}$'), # width of the log-normal particle distribution
                'log_Kzz':([5,15],r'log $K_{zz}$')} # eddy diffusion parameter (atmospheric mixing)
    free_params.update(cloud_props)
    
if GP==True: # add uncertainty scaling
    GP_params={'log_a': ([-1,1], r'$\log\ a$'), # one is enough, will be multiplied with order/det error
               'log_l': ([-3,0], r'$\log\ l$')}
    free_params.update(GP_params)

free_params.update(chemistry)
parameters = Parameters(free_params, constant_params)
cube = np.random.rand(parameters.n_params)
parameters(cube)
params=parameters.params

retrieval=Retrieval(target=brown_dwarf,parameters=parameters,
                    output_name=output,chemistry=chem,
                    cloud_mode=cloud_mode,GP=GP,PT_type=PT_type)
molecules=['H2O','12CO','13CO','HF','H2S','H2(18)O']
retrieval.run_retrieval(N_live_points=Nlive,evidence_tolerance=tol,
                        crosscorr_molecules=molecules,bayes_molecules=molecules)