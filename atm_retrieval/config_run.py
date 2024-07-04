from ast import If
import getpass
import os
import numpy as np
import sys
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

# pass configuration as command line argument
# example: config_run.py 2M0355 freechem graycloud GP
brown_dwarf = sys.argv[1] # options: 2M0355 or 2M1425
chem = sys.argv[2] # options: freechem or equchem
cloud_mode = sys.argv[3] # options: nocloud, graycloud, or Mgcloud
GP = sys.argv[4] # options: GP or noGP
output=f'{brown_dwarf}_{chem}_{cloud_mode}_{GP}' # output folder name

if brown_dwarf=='2M0355': 
    brown_dwarf = Target('2M0355') 
elif brown_dwarf=='2M1425':
    brown_dwarf = Target('2M1425')

if chem=='freechem':
    free_chem=True
elif chem=='equchem':
    free_chem=False

if cloud_mode=='nocloud':
    cloud_mode=None
elif cloud_mode=='graycloud':
    cloud_mode='gray'
elif cloud_mode=='Mgcloud':
    cloud_mode='MgSiO3'
 
if GP=='GP':
    GP=True
elif GP=='noGP':
    GP=False

constant_params={} # add if needed
free_params = {'rv': ([2,20],r'$v_{\rm rad}$'),
               'vsini': ([0,40],r'$v$ sin$i$'),
               'log_g':([3,5],r'log $g$'),
               'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], # limb-darkening coefficient (0-1)

            # P-T profile
            'T1' : ([1000,4000], r'$T_1$'), # bottom of the atmosphere (hotter)
            'T2' : ([0,4000], r'$T_2$'),
            'T3' : ([0,4000], r'$T_3$'),
            'T4' : ([0,4000], r'$T_4$')} # top of atmosphere (cooler)

# if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
if free_chem==False:
    chemistry={'C/O': [(0,1), r'C/O'], 
            'Fe/H': [(-1.5,1.5), r'[Fe/H]'], 
            'C13_12_ratio': [(1e-10,1e-1), r'$\mathrm{^{13}C/^{12}C}$'], 
            'O18_16_ratio': [(1e-10,1e-1), r'$\mathrm{^{18}O/^{16}O}$'], 
            'O17_16_ratio': [(1e-10,1e-1), r'$\mathrm{^{17}O/^{12}O}$']}
    
# if free chemistry, define VMRs
if free_chem==True: 
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
            'log_OH':([-12,-1],r'log OH'),
            'log_CO2':([-12,-1],r'log CO$_2$')}
    
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
    GP_params={'log_a': ([-0.7,0.3], r'$\log\ a$'), # one is enough, will be multipled with order/det error
               'log_l': ([-3.0,-1.0], r'$\log\ l$')}
    free_params.update(GP_params)

free_params.update(chemistry)
parameters = Parameters(free_params, constant_params)
cube = np.random.rand(parameters.n_params)
parameters(cube)
params=parameters.params

retrieval=Retrieval(target=brown_dwarf,parameters=parameters,
                    output_name=output,free_chem=free_chem,
                    cloud_mode=cloud_mode,GP=GP)
retrieval.PMN_run(N_live_points=100,evidence_tolerance=5)
#retrieval.PMN_run(N_live_points=200,evidence_tolerance=0.5)
#only_params=['vsini','log_H2O','log_12CO','log_13CO','T1','T2','T3','T4']
retrieval.evaluate()
