def init_retrieval(target,
                    PT_type,
                    chemistry,
                    Nlive,
                    evtol,
                    cloud_mode=None,
                    use_GP=False,
                    folder_suffix='',*kwargs):
    
    """
    Initialize and configure a spectral retrieval object.

    This function prepares all components required to run an atmospheric
    retrieval, including the target definition, pressure-temperature (P-T)
    parameterization, chemical abundance model, cloud treatment, and sampling
    configuration. It constructs the parameter space, applies target-specific 
    settings, and returns a fully initialized `Retrieval` object.

    Parameters
    ----------
    target : str
        Name of the target object, whose data should exist in a folder of the
        same name, used to initialize a `Target` instance containing 
        observational and instrument metadata.

    PT_type : str
        Parameterization of the atmospheric pressure-temperature profile.
        Supported options include:
        - 'PTknot' : temperature nodes at fixed pressure levels
        - 'PTgrad' : temperature gradients between layers
        - 'PTguillot' : Guillot (2010) analytic radiative equilibrium profile

    chemistry : str
        Chemical abundance model used in the retrieval. Supported options:
        - 'freechem'  : freely retrieved altitude-constant abundances
        - 'equchem'   : equilibrium chemistry
        - 'quequchem' : quenched equilibrium chemistry
        - 'flexequ'   : equilibrium chemistry with scaling factors
        - 'varchem'   : pressure-varying abundances for selected species

    Nlive : int
        Number of live points used in the nested sampling algorithm.

    evtol : float
        Evidence tolerance parameter controlling the stopping criterion
        of the nested sampler.

    cloud_mode : str or None, optional
        Cloud model used in the retrieval. Examples include:
        - 'gray' : gray cloud deck parameterization
        - 'MgSiO3' : physically motivated MgSiO3 cloud model
        - None : no clouds included.

    use_GP : bool, optional
        If True, include Gaussian Process hyperparameters to model correlated
        noise in the data.

    folder_suffix : str, optional
        Additional suffix appended to the retrieval output directory name.


    Returns
    -------
    Retrieval
        A fully configured `Retrieval` object containing the target,
        parameter definitions, species list, and retrieval configuration.
    """

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
    elif getpass.getuser() == "natalie": # when testing from my laptop
        os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    elif 'pRT_input_data_path' in kwargs:
        os.environ['pRT_input_data_path'] = kwargs.get('pRT_input_data_path')
    
    from retrieval import Retrieval
    from parameters import Parameters
    from target import Target

    target = Target(target)

    # initialize parameters dictionaries
    constant_params = {} # add if needed
    free_params = {}
    chemistry_params = {}
    pt_params = {}

    # default setup
    T_upper = 10000
    T_lower = 1000
    Tgrad_lower= 0.0
    log_P_upper = 2
    log_P_lower = -6
    pt_points = 5
    skewed_p_nodes = 1.0 # more pressure points deeper in atm

    offset_equ = 2 # for flexequ
    var_n = 3 # number of pressure points for pressure-variable VMRs
    lowlim = []
    VMR_lowlim = -14
    VMR_uplim = -1
    use_partial_pressure = False # retrieve partial pressure OR VMRs

    disk = False
    gray_cloud_slope = False
    const_efficiency_mode = True
    
    # for running tests
    fix_solar_metall = False
    fix_solar_CO = False
    target_solar_metall = False # penalize metallicity in lnL for freechem
    prior_solar_metall = False # prior for equchem
    fix_logg = False
    fix_PT = False
    fix_all_except_PT = False
    force_cloud = False
    force_Teff = False # penalty on lnL to enforce a certain Teff
    force_phot = False # penalty on lnL to enforce a certain photospheric range

    if target.name in ['2M0355','2M1425','test','test_corr','testsys','SP0829']:
        species_names= ['H2O','12CO','13CO','C18O','C17O','CH4','NH3','HCN','HF','H2(18)O','H2S']
    elif target.name in ['ROXs12A']:
        T_upper = 50000
        species_names= ['H2O','12CO','13CO','HF','H2(18)O','Na','Ti', #'C18O','C17O',
                        'OH','Fe','Sc','CN','Ca','Si','H-','H']#, 'Mg','Cr,'K']
        cloud_mode=None # no clouds at such high temperatures
        disk = True
    elif target.name in ['ROXs12B','ROXs12_onlyB','test_ROXs12B']:
        species_names= ['H2O','12CO','13CO','HF','H2(18)O']
        disk = True
        #target_solar_metall = True # penalize metallicity in lnL

    elif target.name in ['Sorg1X','Sorg20X']:

        species_info = pd.read_csv(os.path.join('species_info_ck.csv'), index_col=0)
        #fix_logg = True
        pt_points = 7
        print(f'Using {pt_points} PT points')
        #fix_PT = True
        print('Fixed PT:',fix_PT)
        const_efficiency_mode = False
        #fix_all_except_PT = True
        if fix_all_except_PT:
            fix_logg = True
            fix_PT = False
        skewed_p_nodes = 1
        print(f'Using skew = {skewed_p_nodes} for P nodes')
        VMR_lowlim = -12
        VMR_uplim = -0.8
        cloud_mode=None
        Tgrad_lower = -0.1 # allow minor temperature inversions
        Tgrad_lower = -0.05 # allow minor temperature inversions
        species_names=['H2O','CH4','C2H6','CO2','C2H2','C2H4','CO','SO2','NH3',
                        'H2S','CS2','OCS','DMS','H2CO','SO','C2H6S2']
    
        # use only main species
        species_names=['H2O','CH4','C2H6','CO2','C2H4','CO','CS2','OCS','DMS']

        #lowlim = ['H2O']
        T_upper = 500
        T_lower = 200
        log_P_upper = 0
        use_GP=False
        offset_equ = 5
        vary_species = []
        if chemistry=='varchem':
            vary_species = ['H2O'] # can expand on this
            #vary_species = species_names # retrieve altitude-dependent VMRs for all
        fixed_species = [s for s in species_names if s not in vary_species]
        
        # penalty on Teff
        #force_Teff = True
        Teff_ref = 280
        Teff_ref_err = 5

        use_partial_pressure = True
        partial_P_lower = -12
        partial_P_upper = -1

        # force photosphere into certain pressure region
        force_phot = False
        log_phot_upper = -1
        log_phot_lower = -2
        phot_fraction = 0.8
        phot_alpha = 200 

    ########## create parameters dict ###############

    if target.instrument=='CRIRES':
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        free_params.update({'rv': ([-20,50],r'$v_{\rm rad}$'),
                            'vsini': ([0,40],r'$v$ sin$i$'),
                            'epsilon_limb': ([0,1], r'$\epsilon_\mathrm{limb}$')})
        if fix_logg and target.name in ['ROXs12B','ROXs12_onlyB']:
            constant_params['log_g'] = 4.0
        else:
            free_params.update({'log_g':([3,6],r'log $g$')})

        if target.name in ['ROXs12A']:
            free_params.update({'log_g': ({'type': 'gaussian', 'mu': 4.1, 'sigma': 0.15,
                                                'bounds': (3.6, 4.6)}, r'$\log g$')})
            free_params.update({'epsilon_limb': ({'type': 'gaussian', 'mu': 0.4, 'sigma': 0.1,
                                                'bounds': (0,1)}, r'$\epsilon_\mathrm{limb}$')})

        elif target.name in ['ROXs12B','ROXs12_onlyB','test_ROXs12B']:
            # gaussian prior on logg
            free_params.update({'log_g': ({'type': 'gaussian', 'mu': 4.0, 'sigma': 0.1,
                                            'bounds': (3.5, 4.5)}, r'$\log g$')})

        # contribution of BD to total spectrum, linear function of wavelength
        if target.primary_label==False: #+star
            free_params.update({#'log_phi_k': ([-3,-4], r'log $\phi_k$'), # slope 
                                #'phi_d': ([0.4,0.9], r'$\phi_d$'),# intercept
                                'phi_secondary': ([0.5,0.9], r'$\phi_\mathrm{BD}$')}) # retrieve as only 1 param
            free_params.update({'star_fwhm': ([0,100], r'FWHM$_\mathrm{s}$')}) # broadening of starlight [km/s]

    elif target.name in ['Sorg1X','Sorg20X']:
        if fix_logg:
            constant_params['log_g'] = 3.09
        else:
            free_params.update({'log_g':([2.5,3.5],r'log $g$')})

    if force_Teff:
        constant_params['Teff_ref'] = Teff_ref
        constant_params['Teff_ref_err'] = Teff_ref_err

    constant_params['skewed_p_nodes'] = skewed_p_nodes
    constant_params['fix_all_except_PT'] = fix_all_except_PT
    constant_params['const_efficiency_mode'] = const_efficiency_mode
    if const_efficiency_mode:
        constant_params['sampling_efficiency'] = 0.5
    else:
        constant_params['sampling_efficiency'] = 0.5
    print(f"Constant efficiency mode = {const_efficiency_mode}; sampling efficiency = {constant_params['sampling_efficiency']}")

    if use_partial_pressure and chemistry=='varchem':
        # middle pressure, only works for var_n = 3
        free_params['log_P_1'] = ([0,-2],rf'log $p_1$')

    constant_params['log_P_upper'] = log_P_upper # top of atm
    constant_params['log_P_lower'] = log_P_lower # bottom of atm

    if force_phot:
        constant_params['log_phot_upper'] = log_phot_upper
        constant_params['log_phot_lower'] = log_phot_lower
        constant_params['phot_fraction'] = phot_fraction
        constant_params['phot_alpha'] = phot_alpha 

    if target_solar_metall:
        constant_params['target_solar_metall'] = True

    constant_params['fix_PT'] = fix_PT
    pt_params={}
    if PT_type=='PTknot' and fix_PT==False: 
        for n in range(pt_points):
            pt_params[f'T{n}']=([0,T_upper],rf'$T_{n}$') # T0 = bottom of atmosphere

    # not working correctly yet
    elif PT_type in ['PTgrad','PTgradvar'] and fix_PT==False:
        for n in range(pt_points):
            pt_params[f'dlnT_dlnP_{n}']=([Tgrad_lower,0.4],rf'$\nabla T_{n}$')
        pt_params['T0']= ([T_lower,T_upper], r'$T_0$') # T0 = bottom of atmosphere
        if PT_type=='PTgradvar':
            pt_params['log_P_RCB'] = ([log_P_lower,log_P_upper], r'$P_\mathrm{RCB}$')
            pt_params['log_delta_P'] = ([-1,0], r'$\Delta P$')
       
    elif PT_type=='PTguillot' and fix_PT==False: # just for Sorg1/20X for now
        pt_params['T_int']= ([20,200], r'$T_\mathrm{int}$') # internal
        pt_params['T_equ']= ([200,400], r'$T_\mathrm{equ}$') # blackbody
        pt_params['log_k_IR']= ([-3,0.5], r'log $\kappa_\mathrm{IR}$')
        pt_params['log_gamma']= ([-2,1], r'log $\gamma$')
    
    free_params.update(pt_params)

    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    if chemistry in ['equchem','quequchem','flexequ']:
        chemistry_params={}
        if chemistry in ['equchem','quequchem']:
            if fix_solar_metall or chemistry=='flexequ':
                constant_params['Fe/H'] = 0.0 # solar
            elif prior_solar_metall:
                chemistry_params.update({'Fe/H': ({'type': 'gaussian', 'mu': 0.0, 'sigma': 0.1,
                                            'bounds': (-0.5, 0.5)}, r'[Fe/H]')})
            else:
                chemistry_params.update({'Fe/H': ([-1.,1.], r'[Fe/H]')})
            if fix_solar_CO or chemistry=='flexequ':
                constant_params['C/O'] = 0.59 # solar, Asplund 2021
            else:
                chemistry_params.update({'C/O':([0.1,1.], r'C/O')})

        if chemistry=='flexequ': # vary equchem abundances by constant factor
            for species in species_names:
                if species in ['C2H6','CO2','C2H4','CO','H2S','DMS']:
                    chemistry_params[f'log_a_{species}'] = ([0,100],fr'log $\alpha$ {species_info.loc[species,"mathtext_name"]}')
                elif species not in ['13CO','C17O','C18O','H2(18)O']: # isotopes etrieved separately
                    chemistry_params[f'log_a_{species}'] = ([-offset_equ,offset_equ],fr'log $\alpha$ {species_info.loc[species,"mathtext_name"]}')

        if target.instrument=='CRIRES': # only for high-res
            if '13CO' in species_names:
                chemistry_params['log_C12_13_ratio'] = ([1,6], r'log $\mathrm{^{12}CO/^{13}CO}$')
            if 'C18O' in species_names:
                chemistry_params['log_O16_18_ratio'] = ([1,6], r'log $\mathrm{C^{16}O/C^{18}O}$')
            if 'H2(18)O' in species_names:
                chemistry_params['log_H2O16_18_ratio'] = ([1,6], r'log $\mathrm{H_2^{16}O/H_2^{18}O}$')
            if 'C17O' in species_names: 
                chemistry_params['log_O16_17_ratio'] = ([1,6], r'log $\mathrm{C^{16}O/C^{17}O}$')
            
        if chemistry=='quequchem': # quenched equilibrium chemistry_params
            chemistry_params.update({#'log_Pqu_CO_CH4': ([log_P_lower,log_P_upper], r'log P$_{qu}$(CO,CH$_4$,H$_2$O)'),
                            #'log_Pqu_NH3': ([log_P_lower,log_P_upper], r'log P$_{qu}$(NH$_3$)'),
                            #'log_Pqu_HCN': ([log_P_lower,log_P_upper], r'log P$_{qu}$(HCN)')
                            'log_Pqu_H2O_OH_O': ([log_P_lower,log_P_upper], r'log P$_{qu}$(H$_2$O,OH,O)')
                            })
        
    # if free chemistry_params, define VMRs
    elif chemistry=='freechem': 
        if use_partial_pressure:
            for species_i in species_names:
                chemistry_params[f"log_p_{species_i}"]=([partial_P_lower,partial_P_upper],rf"log $p$ {species_info.loc[species_i,'mathtext_name']}")
        else:
            for species_i in species_names:
                if species_i in lowlim:
                    print('Using lower limit for',species_i)
                    chemistry_params[f"log_{species_i}"]=([-2,VMR_uplim],rf"log {species_info.loc[species_i,'mathtext_name']}")
                else:
                    chemistry_params[f"log_{species_i}"]=([VMR_lowlim,VMR_uplim],rf"log {species_info.loc[species_i,'mathtext_name']}")
            if 'H-' in species_names:
                chemistry_params[f"log_e-"]=([VMR_lowlim,VMR_uplim],"log e$^{-}$")
                
    elif chemistry=='varchem':
        for species_i in vary_species:
            for i in range(var_n):
                if use_partial_pressure:
                    chemistry_params[f"log_p_{species_i}_{i}"]=([partial_P_lower,partial_P_upper],rf"log $p$ {species_info.loc[species_i,'mathtext_name']} {i}")
                else:
                    chemistry_params[f"log_{species_i}_{i}"]=([VMR_lowlim,VMR_uplim],rf"log {species_info.loc[species_i,'mathtext_name']} {i}")
        for species_i in fixed_species:
            if use_partial_pressure:
                chemistry_params[f"log_p_{species_i}"]=([partial_P_lower,partial_P_upper],rf"log $p$ {species_info.loc[species_i,'mathtext_name']}")
            else:
                chemistry_params[f"log_{species_i}"]=([VMR_lowlim,VMR_uplim],rf"log {species_info.loc[species_i,'mathtext_name']}")

    if use_partial_pressure: # also for H2 and He
        chemistry_params[f"log_p_H2"]=([-1,-0.05],rf"log $p$ H$_2$")
        chemistry_params[f"log_p_He"]=([-2,-0.2],rf"log $p$ He")  
        
    if disk==True: # retrieve veiling factor as linear function
        free_params.update({'phi_disk': ([0,1], r'$r_k(\lambda_\mathrm{mid}$'),#r'$\phi_\mathrm{disk}$'), # contribution 
                            'T_disk': ([300,1500], r'$T_\mathrm{disk}$')}) # temperature
        #free_params.update({'log_k_rk': ([-6,-2], r'log $k_\mathrm{rk}$'), # slope log 1/nm
                            #'d_rk': ([0,5], r'$d_\mathrm{rk}$')}) # intercept (rk at bluest wl)

    if cloud_mode=='gray': # gray cloud deck
        if force_cloud:
                cloud_params={'log_opa_base_gray': ([0,2], r'log $\kappa_{\mathrm{cl},0}$'), # opacity at cloud base
                        'log_P_base_gray': ([-1,0.5], r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
                        'fsed_gray': ([0,10], r'$f_\mathrm{sed}$')} # sedimentation parameter for particles
        else:
            cloud_params={'log_opa_base_gray': ([-10,3], r'log $\kappa_{\mathrm{cl},0}$'), # opacity at cloud base
                        'log_P_base_gray': ([-6,3], r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
                        'fsed_gray': ([0,20], r'$f_\mathrm{sed}$')} # sedimentation parameter for particles
        if gray_cloud_slope:
            cloud_params['cloud_slope'] = ([-4,0], r'$\gamma_{\mathrm{cl}}$')
        free_params.update(cloud_params)

    if cloud_mode=='MgSiO3':
        cloud_params={'fsed': ([0,20], r'$f_\mathrm{sed}$'), # sedimentation parameter for particles
                    'sigma_lnorm': ([0.8,1.5], r'$\sigma_{l,norm}$'), # width of the log-normal particle distribution
                    'log_Kzz':([5,15],r'log $K_{zz}$')} # eddy diffusion parameter (atmospheric mixing)
        free_params.update(cloud_params)
        
    if use_GP==True: # uncertainty scaling through Gaussian processes
        GP_params={'log_a': ([-1,1], r'$\log\ a$'),
                'log_l': ([-3,1], r'$\log\ l$')}
        free_params.update(GP_params)

    if fix_all_except_PT:
        chemistry_params = {}
    free_params.update(chemistry_params)
    parameters = Parameters(free_params, constant_params)
    cube = np.random.rand(parameters.n_params)
    parameters(cube)

    retrieval_object = Retrieval(target=target,
                        parameters=parameters,
                        species_names=species_names,
                        Nlive=Nlive,evtol=evtol,
                        chemistry=chemistry,
                        PT_type=PT_type,
                        cloud_mode=cloud_mode,
                        use_GP=use_GP,
                        use_partial_pressure=use_partial_pressure,
                        folder_suffix=folder_suffix)

    return retrieval_object

if __name__ == '__main__':
    from datetime import datetime
    import sys, shutil, os
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
    from datetime import datetime

    class SuppressOutput:
        def __enter__(self):
            if rank != 0:
                # Redirect both stdout and stderr to devnull
                self._stdout = os.dup(1)
                self._stderr = os.dup(2)
                self.devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.devnull, 1)
                os.dup2(self.devnull, 2)
        def __exit__(self, exc_type, exc_val, exc_tb):
            if rank != 0:
                # Restore original stdout and stderr
                os.dup2(self._stdout, 1)
                os.dup2(self._stderr, 2)
                os.close(self.devnull)

    with SuppressOutput(): # print only once when running parallel

        start = datetime.now()
        print("Start time:", start.strftime("%Y-%m-%d %H:%M:%S"))

        # pass configuration as command line argument
        # example: config_run.py 2M0355 freechem PTgrad 200 5
        target = sys.argv[1] # 2M0355 / 2M1425 / test
        chemistry = sys.argv[2] # freechem / equchem / quequchem / flexequ / varchem
        PT_type = sys.argv[3] # PTknot / PTgrad / PTguillot
        Nlive=int(sys.argv[4]) # number of live points (integer)
        evtol=float(sys.argv[5]) # evidence tolerance (float)
        folder_suffix = sys.argv[6] if len(sys.argv)>6 else '' # for note
        bayes_species=sys.argv[7] if len(sys.argv)>7 else None # bayes evidence retrievals on specified species

        retrieval_object = init_retrieval(target=target,PT_type=PT_type,
                                 chemistry=chemistry,Nlive=Nlive,
                                evtol=evtol,folder_suffix=folder_suffix)

        if rank == 0:  # only first process saves a copy of config file
            if not any(fname.startswith("config_run_") for fname in os.listdir(retrieval_object.output_dir)):
                timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
                shutil.copy(__file__, os.path.join(retrieval_object.output_dir, f"config_run_{timestamp}.py"))

        retrieval_object.run_retrieval(bayes_species=bayes_species)

        end = datetime.now()
        print("End time:  ", end.strftime("%Y-%m-%d %H:%M:%S"))
        dt_minutes = (end - start).total_seconds() / 60
        print(f"Elapsed time: {dt_minutes:.2f} minutes")