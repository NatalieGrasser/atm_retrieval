
def init_retrieval(target,PT_type,chem,Nlive,evtol,
                    cloud_mode='gray',GP=True,folder_suffix=''):

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
    constant_params={} # add if needed
    free_params = {}
    T_upper = 10000
    T_lower = 1000
    Tgrad_lower= 0.0
    log_P_upper = 2
    log_P_lower = -6
    n_grad = 5
    offset_equ = 2 # for flexequ
    chemistry = {}
    pt_params = {}
    var_n = 3 # number of pressure points for pressure-variable VMRs
    disk = False
    partial_pressure = False # retrieve partial pressure OR VMRs
    fix_solar_metall_CO = False
    target_solar_metall = True # penalize metallicity in lnL
    gray_cloud_slope = False

    if target.instrument=='CRIRES':
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        free_params.update({'rv': ([-20,50],r'$v_{\rm rad}$'),
                            'vsini': ([0,40],r'$v$ sin$i$'),
                            'log_g':([3,6],r'log $g$'),
                            'epsilon_limb': ([0.2,1], r'$\epsilon_\mathrm{limb}$')})
    elif target.instrument=='LIFE':
        species_info = pd.read_csv(os.path.join('species_info_ck.csv'), index_col=0)
        #free_params.update({'log_g':([2.5,3.5],r'log $g$')})
        #free_params.update({'log_g':([2.9,3.3],r'log $g$')})
        free_params.update({'log_g':([3.089,3.091],r'log $g$')})
        #constant_params['log_g'] = 3.09

    if target.name in ['2M0355','2M1425','test','test_corr','testsys','SP0829']:
        species_names= ['H2O','12CO','13CO','C18O','C17O','CH4','NH3','HCN','HF','H2(18)O','H2S']
    elif target.name in ['ROXs12A']:
        #species_names= ['H2O','12CO','13CO','HF','H2(18)O','Na','Ti','OH','Fe','Sc','CN','Ca','Si',
                        #'Ba','Mg','Cr','CH','V']#,'VO','SiO','TiO','MgO', 'SH', 'ScH']
        species_names= ['H2O','12CO','13CO','HF','H2(18)O','Na','Ti','OH','Fe','Sc','CN','Ca','Si',
                        'Mg','Cr','K']#,'H']
        cloud_mode=None # no clouds at such high temperatures
        disk = True
    elif target.name in ['ROXs12B','ROXs12_onlyB']:
        species_names= ['H2O','12CO','13CO', 'HF','H2(18)O']#,'Ca','FeH','VO','TiO']#,'Na','Ti']
        cloud_mode=None
        #species_names= ['H2O','12CO','13CO', 'HF','H2(18)O']
        #species_names= ['H2O','12CO','13CO','HF','H2(18)O','Na','Ti','OH','Fe','Sc','CN','Ca','Si',
                       #'Mg','Cr','K','H']

    if target.name in ['ROXs12B','ROXs12_onlyB','testsys']:
        disk = True
        if target.name in ['ROXs12B','ROXs12_onlyB']:
            log_P_upper = 1
            log_P_lower = -4
        # contribution of BD to total spectrum, linear function of wavelength
        free_params.update({#'log_phi_k': ([-3,-4], r'log $\phi_k$'), # slope 
                            #'phi_d': ([0.4,0.9], r'$\phi_d$'),# intercept
                            'phi_secondary': ([0.5,0.9], r'$\phi_\mathrm{BD}$'), # retrieve as only 1 param
                            'star_fwhm': ([0,100], r'FWHM$_\mathrm{s}$')}) # broadening of starlight [km/s]

    elif target.name in ['Sorg1X','Sorg20X']:
        cloud_mode=None
        #PT_type='PTknot'
        Tgrad_lower = -0.1 # allow minor temperature inversions
        species_names=['H2O','CH4','C2H6','CO2','C2H2','C2H4','CO','SO2','NH3',
                        'H2S','CS2','OCS','DMS','H2CO','SO','C2H6S2']
        species_names=['H2O','CH4','C2H6','CO2','NH3','C2H4','CO','H2S','DMS']
        T_upper = 500
        T_lower = 100
        n_grad = 5
        log_P_upper = 0
        GP=True
        offset_equ = 5
        #vary_species = species_names # retrieve altitude-dependent VMRs for all
        vary_species = ['H2O'] # can expand on this
        fixed_species = [s for s in species_names if s not in vary_species]
        Teff_ref = 280
        Teff_ref_err = 20
        partial_pressure = False
        #partial_P_lower = -10
        partial_P_lower = -8
        partial_P_upper = -0.5
        # force emission contribution into certain pressure region
        log_emcont_upper = -1
        log_emcont_lower = -2
        emcont_fraction = 0.8
        emcont_alpha = 200 

    if 'Teff_ref' in locals():
        constant_params['Teff_ref'] = Teff_ref
        constant_params['Teff_ref_err'] = Teff_ref_err

    if partial_pressure and chem=='varchem':
        # middle pressure, only works for var_n = 3
        free_params['log_P_1'] = ([0,-2],rf'log $p_1$')

    constant_params['log_P_upper'] = log_P_upper # top of atm
    constant_params['log_P_lower'] = log_P_lower # bottom of atm

    if 'emcont_fraction' in locals():
        constant_params['log_emcont_upper'] = log_emcont_upper
        constant_params['log_emcont_lower'] = log_emcont_lower
        constant_params['emcont_fraction'] = emcont_fraction
        constant_params['emcont_alpha'] = emcont_alpha 

    if target_solar_metall:
        constant_params['target_solar_metall'] = True

    if PT_type=='PTknot': 
        n_knots = 5
        for n in range(n_knots):
            pt_params[f'T{n}']=([0,T_upper],rf'$T_{n}$') # T0 = bottom of atmosphere

    elif PT_type in ['PTgrad','PTgradvar']:
        for n in range(n_grad):
            pt_params[f'dlnT_dlnP_{n}']=([Tgrad_lower,0.4],rf'$\nabla T_{n}$')
        pt_params['T0']= ([T_lower,T_upper], r'$T_0$') # T0 = bottom of atmosphere
        if PT_type=='PTgradvar':
            #constant_params['log_P_0'] = log_P_upper # const, at bottom of atmosphere
            #constant_params[f'log_P_{int(n_grad-1)}'] = log_P_lower # top of atm
            #for n in np.linspace(1,n_grad,n_grad,dtype=int)[:-2]: # 0 and last int not included
                #pt_params[f'log_P_{n}']=([log_P_lower,log_P_upper],rf'log $P_{n}$')
            #n_array = np.linspace(1,n_grad,n_grad,dtype=int)[:-2]
            #central_n = int(np.median(n_array))
            pt_params['log_P_RCB'] = ([log_P_lower,log_P_upper], r'$P_\mathrm{RCB}$')
            pt_params['log_delta_P'] = ([-1,0], r'$\Delta P$')

    elif PT_type=='PTguillot': # just for K2-18b for now
        pt_params['T_int']= ([20,200], r'$T_\mathrm{int}$') # internal
        pt_params['T_equ']= ([200,400], r'$T_\mathrm{equ}$') # blackbody
        pt_params['log_k_IR']= ([-3,0.5], r'log $\kappa_\mathrm{IR}$')
        pt_params['log_gamma']= ([-2,1], r'log $\gamma$')
    
    free_params.update(pt_params)

    if partial_pressure: # also for H2 and He
        chemistry[f"log_p_H2"]=([-1,0],rf"log $p$ H$_2$")
        chemistry[f"log_p_He"]=([-2,-0.2],rf"log $p$ He")

    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    if chem in ['equchem','quequchem','flexequ']:

        if chem in ['equchem','quequchem'] and fix_solar_metall_CO==False:
            chemistry={'C/O':([0.1,1.], r'C/O'), 
                        'Fe/H': ([-1.,1.], r'[Fe/H]')}
        elif (chem in ['equchem','quequchem'] and fix_solar_metall_CO) or (chem=='flexequ'):
            constant_params['C/O'] = 0.59 # solar, Asplund 2021
            constant_params['Fe/H'] = 0.0 # solar
        if chem=='flexequ': # vary equchem abundances by constant factor
            for species in species_names:
                if species in ['C2H6','CO2','C2H4','CO','H2S','DMS']:
                    chemistry[f'log_a_{species}'] = ([0,100],fr'log $\alpha$ {species_info.loc[species,"mathtext_name"]}')
                elif species not in ['13CO','C17O','C18O','H2(18)O']: # retrieved separately
                    chemistry[f'log_a_{species}'] = ([-offset_equ,offset_equ],fr'log $\alpha$ {species_info.loc[species,"mathtext_name"]}')

        if target.instrument=='CRIRES': # only for high-res
            if '13CO' in species_names:
                chemistry['log_C12_13_ratio'] = ([1,6], r'log $\mathrm{^{12}CO/^{13}CO}$')
            if 'C18O' in species_names:
                chemistry['log_O16_18_ratio'] = ([1,6], r'log $\mathrm{C^{16}O/C^{18}O}$')
            if 'H2(18)O' in species_names:
                chemistry['log_H2O16_18_ratio'] = ([1,6], r'log $\mathrm{H_2^{16}O/H_2^{18}O}$')
            if 'C17O' in species_names: 
                chemistry['log_O16_17_ratio'] = ([1,6], r'log $\mathrm{C^{16}O/C^{17}O}$')
            
        if chem=='quequchem': # quenched equilibrium chemistry
            chemistry.update({'log_Pqu_CO_CH4': ([-6,2], r'log P$_{qu}$(CO,CH$_4$,H$_2$O)'),
                            'log_Pqu_NH3': ([-6,2], r'log P$_{qu}$(NH$_3$)'),
                            'log_Pqu_HCN': ([-6,2], r'log P$_{qu}$(HCN)')})
        
    # if free chemistry, define VMRs
    elif chem=='freechem': 
        if partial_pressure:
            for species_i in species_names:
                chemistry[f"log_p_{species_i}"]=([partial_P_lower,partial_P_upper],rf"log $p$ {species_info.loc[species_i,'mathtext_name']}")
        else:
            for species_i in species_names:
                chemistry[f"log_{species_i}"]=([-14,-1],rf"log {species_info.loc[species_i,'mathtext_name']}")
    elif chem=='varchem':
        for species_i in vary_species:
            for i in range(var_n):
                if partial_pressure:
                    chemistry[f"log_p_{species_i}_{i}"]=([partial_P_lower,partial_P_upper],rf"log $p$ {species_info.loc[species_i,'mathtext_name']} {i}")
                else:
                    #chemistry[f"log_{species_i}_{i}"]=([-14,-1],rf"log {species_info.loc[species_i,'mathtext_name']}")
                    # place stronger prior on H2O abundance in photosphere
                    #if i in [1,2]:
                        #chemistry[f"log_{species_i}_{i}"]=([-2,-1],rf"log {species_info.loc[species_i,'mathtext_name']}")
                    #else:
                    chemistry[f"log_{species_i}_{i}"]=([-14,-1],rf"log {species_info.loc[species_i,'mathtext_name']} {i}")
        for species_i in fixed_species:
            if partial_pressure:
                chemistry[f"log_p_{species_i}"]=([partial_P_lower,partial_P_upper],rf"log $p$ {species_info.loc[species_i,'mathtext_name']}")
            else:
                chemistry[f"log_{species_i}"]=([-14,-1],rf"log {species_info.loc[species_i,'mathtext_name']}")
        
    if disk==True: # retrieve veiling factor as linear function
        #free_params.update({'phi_disk': ([0,1], r'log $\phi_\mathrm{disk}$'), # contribution 
                            #'T_disk': ([300,1200], r'$T_\mathrm{disk}$')}) # temperature
        free_params.update({'log_k_rk': ([-6,-2], r'log $k_\mathrm{rk}$'), # slope log 1/nm
                            'd_rk': ([0,5], r'$d_\mathrm{rk}$')}) # intercept (rk at bluest wl)

    if cloud_mode=='gray':
        cloud_props={'log_opa_base_gray': ([-10,3], r'log $\kappa_{\mathrm{cl},0}$'),  
                    'log_P_base_gray': ([-6,3], r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
                    'fsed_gray': ([0,20], r'$f_\mathrm{sed}$')} # sedimentation parameter for particles
        if gray_cloud_slope:
            cloud_props['cloud_slope'] = ([-4,0], r'$\gamma_{\mathrm{cl}}$')
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
                        Nlive=Nlive,evtol=evtol,chemistry=chem,PT_type=PT_type,
                        cloud_mode=cloud_mode,GP=GP,partialP=partial_pressure,
                        folder_suffix=folder_suffix)

    return retrieval

if __name__ == '__main__':
    from datetime import datetime
    import sys, shutil, os
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # pass configuration as command line argument
    # example: config_run.py 2M0355 freechem PTgrad 200 5
    target = sys.argv[1] # 2M0355 / 2M1425 / test
    chem = sys.argv[2] # freechem / equchem / quequchem
    PT_type = sys.argv[3] # PTknot / PTgrad
    Nlive=int(sys.argv[4]) # number of live points (integer)
    evtol=float(sys.argv[5]) # evidence tolerance (float)
    folder_suffix = sys.argv[6] if len(sys.argv)>6 else '' # for note
    bayes_species=sys.argv[7] if len(sys.argv)>7 else None # bayes evidence retrievals on specified species

    retrieval=init_retrieval(target=target,PT_type=PT_type,chem=chem,Nlive=Nlive,
                            evtol=evtol,folder_suffix=folder_suffix)

    if rank == 0:  # only first process saves a copy of config file
        if not any(fname.startswith("config_run_") for fname in os.listdir(retrieval.output_dir)):
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
            shutil.copy(__file__, os.path.join(retrieval.output_dir, f"config_run_{timestamp}.py"))

    retrieval.run_retrieval(bayes_species=bayes_species)