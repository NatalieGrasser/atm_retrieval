
def init_retrieval(target,PT_type,chem,Nlive,evtol,cloud_mode='gray',GP=True):

    import getpass
    import os
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice") # ignore warning for empty orders
    warnings.filterwarnings("ignore", message="All-NaN slice encountered") # ignore warning for empty orders
    os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs, important for MPI

    if getpass.getuser() == "grasser": # when running from LEM
        os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"
        from mpi4py import MPI 
        comm = MPI.COMM_WORLD # important for MPI
        rank = comm.Get_rank() # important for MPI
        import matplotlib
        matplotlib.use('Agg') # disable interactive plotting
    elif getpass.getuser() == "natalie": # when testing from my laptop
        os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    from retrieval import Retrieval
    from parameters import Parameters
    from target import Target

    target = Target(target)
    species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)

    if target.name in ['2M0355','2M1425','test','test_corr','testsys']:
        species_names= ['H2O','12CO','13CO','C18O','C17O','CH4','NH3','HCN','HF','H2(18)O','H2S']
    elif target.name in ['ROXs12A']:
        species_names= ['H2O','12CO','13CO','C18O','C17O','HF','H2(18)O','Na','Ti',
                        'OH','CN','Fe','Sc','Si','Mg','V','K','Al','Mn','TiH','Cs','Ni','Rb']
    elif target.name in ['ROXs12B']:
        species_names= ['H2O','12CO','13CO','C18O','C17O','CH4','NH3','HCN','HF','H2(18)O','H2S']     

    constant_params={} # add if needed
    free_params = {'rv': ([-20,20],r'$v_{\rm rad}$'),
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
                'T0': ([1000,6000], r'$T_0$')} # at bottom of atmosphere
        free_params.update(pt_params)

    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    if chem in ['equchem','quequchem']:
        chemistry={'C/O':([0,1], r'C/O'), 
                'Fe/H': ([-1.5,1.5], r'[Fe/H]'), 
                'log_C12_13_ratio': ([1,12], r'log $\mathrm{^{12}C/^{13}C}$'), 
                'log_O16_18_ratio': ([1,12], r'log $\mathrm{^{16}O/^{18}O}$'), 
                'log_O16_17_ratio': ([1,12], r'log $\mathrm{^{16}O/^{17}O}$')}
            
        if chem=='quequchem': # quenched equilibrium chemistry
            chemistry.update({'log_Pqu_CO_CH4': ([-6,2], r'log P$_{qu}$(CO,CH$_4$,H$_2$O)'),
                            'log_Pqu_NH3': ([-6,2], r'log P$_{qu}$(NH$_3$)'),
                            'log_Pqu_HCN': ([-6,2], r'log P$_{qu}$(HCN)')})
        
    # if free chemistry, define VMRs
    if chem=='freechem': 
        chemistry={}
        for species_i in species_names:
            chemistry[species_i]=([-12,-1],rf"log {species_info.loc[species_i,'mathtext_name']}")
        
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
        GP_params={'log_a': ([-1,1], r'$\log\ a$'), # one is enough, will be multipled with order/det error
                'log_l': ([-3,1], r'$\log\ l$')}
        free_params.update(GP_params)

    free_params.update(chemistry)
    parameters = Parameters(free_params, constant_params)
    cube = np.random.rand(parameters.n_params)
    parameters(cube)

    retrieval=Retrieval(target=target,parameters=parameters,species_names=species_names,
                        Nlive=Nlive,evtol=evtol,chemistry=chem,PT_type=PT_type)

    return retrieval

if __name__ == '__main__':
    import sys

    # pass configuration as command line argument
    # example: config_run.py 2M0355 freechem PTgrad 200 5
    target = sys.argv[1] # 2M0355 / 2M1425 / test
    chem = sys.argv[2] # freechem / equchem / quequchem
    PT_type = sys.argv[3] # PTknot / PTgrad
    Nlive=int(sys.argv[4]) # number of live points (integer)
    evtol=float(sys.argv[5]) # evidence tolerance (float)
    bayes_molecules=sys.argv[6] if len(sys.argv)>6 else None # bayes evidence retrievals on specified species

    retrieval=init_retrieval(target=target,PT_type=PT_type,chem=chem,Nlive=Nlive,evtol=evtol)
    retrieval.run_retrieval(bayes_molecules=bayes_molecules)
