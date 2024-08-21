# run this to make comparison plots of 2M0355 and 2M1425 retrievals (must have same settings)

import getpass
import os
import numpy as np
import sys

if getpass.getuser() == "grasser": # when running from LEM
    os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"
    from atm_retrieval.target import Target
    from atm_retrieval.retrieval import Retrieval
    from atm_retrieval.parameters import Parameters
    import atm_retrieval.figures as figs
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    from target import Target
    from retrieval import Retrieval
    from parameters import Parameters
    import figures as figs

# pass configuration as command line argument
# template: python3 compare_retrievals.py BD1 chem1 PT1 Nlive1 tol1 BD2 chem2 PT2 Nlive2 tol2
# example: python3 compare_retrievals.py 2M0355 freechem PTgrad 300 0.5 2M1425 freechem PTgrad 300 0.5
# PT, Nlive, tol must be same for both
BD1=sys.argv[1]
chem1 = sys.argv[2]
PT1 = sys.argv[3]
Nlive1=int(sys.argv[4])
tol1=float(sys.argv[5])

BD2=sys.argv[6]
chem2 = sys.argv[7]
PT2 = sys.argv[8]
Nlive2=int(sys.argv[9])
tol2=float(sys.argv[10])

BD3=None
# example: python3 compare_retrievals.py 2M0355 freechem PTgrad 300 0.5 2M0355 equchem PTgrad 50 10.0 2M0355 quequchem PTgrad 55 10.0
if len(sys.argv)>11:
    BD3=sys.argv[11]
    chem3 = sys.argv[12]
    PT3 = sys.argv[13]
    Nlive3=int(sys.argv[14])
    tol3=float(sys.argv[15])
    
def init_retrieval(brown_dwarf='2M0355',PT_type='PTgrad',chem='freechem',Nlive=400,tol=0.5,cloud_mode='gray',GP=True):

    brown_dwarf = Target(brown_dwarf)
    output=f'{chem}_{PT_type}_N{Nlive}_ev{tol}' # output folder name

    constant_params={} # add if needed
    free_params = {'rv': ([2,20],r'$v_{\rm rad}$'),
                'vsini': ([0,40],r'$v$ sin$i$'),
                'log_g':([3,5],r'log $g$'),
                'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$']} # limb-darkening coefficient (0-1)

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
                'log_C12_13_ratio': ([1,12], r'log $\mathrm{^{12}C/^{13}C}$'), 
                'log_O16_18_ratio': ([1,12], r'log $\mathrm{^{16}O/^{18}O}$'), 
                'log_O16_17_ratio': ([1,12], r'log $\mathrm{^{16}O/^{17}O}$')}
        
    if chem=='quequchem': # quenched equilibrium chemistry
        chemistry={'C/O':([0,1], r'C/O'), 
                'Fe/H': ([-1.5,1.5], r'[Fe/H]'), 
                'log_C12_13_ratio': ([1,12], r'log $\mathrm{^{12}C/^{13}C}$'), 
                'log_O16_18_ratio': ([1,12], r'log $\mathrm{^{16}O/^{18}O}$'), 
                'log_O16_17_ratio': ([1,12], r'log $\mathrm{^{16}O/^{17}O}$'),
                'log_Pqu_CO_CH4': ([-6,2], r'log P$_{qu}$(CO,CH$_4$,H$_2$O)'),
                'log_Pqu_NH3': ([-6,2], r'log P$_{qu}$(NH$_3$)'),
                'log_Pqu_HCN': ([-6,2], r'log P$_{qu}$(HCN)')}  
        
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
                'log_H2S':([-12,-1],r'log H$_2$S')}
        
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
                'log_l': ([-3,0], r'$\log\ l$')}
        free_params.update(GP_params)

    free_params.update(chemistry)
    parameters = Parameters(free_params, constant_params)
    cube = np.random.rand(parameters.n_params)
    parameters(cube)

    output=f'{chem}_{PT_type}_N{Nlive}_ev{tol}' # output folder name
    retrieval=Retrieval(target=brown_dwarf,parameters=parameters,
                    output_name=output,chemistry=chem,PT_type=PT_type)

    return retrieval

retrieval=init_retrieval(brown_dwarf=BD1,PT_type=PT1,chem=chem1,Nlive=Nlive1,tol=tol1)
retrieval.evaluate(makefigs=False)

retrieval2=init_retrieval(brown_dwarf=BD2,PT_type=PT2,chem=chem2,Nlive=Nlive2,tol=tol2)
if BD1==BD2:
    # give it a different color
    retrieval2.color1='mediumpurple' # color of retrieval output
    retrieval2.color2='blueviolet' # color of residuals
    retrieval2.color3='mediumorchid'
retrieval2.evaluate(makefigs=False)

if BD3!=None: # compare freechem, equchem, quequchem of same object
    retrieval3=init_retrieval(brown_dwarf=BD3,PT_type=PT3,chem=chem3,Nlive=Nlive3,tol=tol3)
    retrieval3.color1='limegreen' # color of retrieval output
    retrieval3.color2='forestgreen' # color of residuals
    retrieval3.color3='yellowgreen'
    retrieval3.evaluate(makefigs=False)
    figs.compare_retrievals(retrieval,retrieval2,retrieval_object3=retrieval3)
    molecules=['H2','He','H2O','H2(18)O','12CO','13CO','CH4','NH3']
    figs.VMR_plot(retrieval,retrieval_object2=retrieval2,retrieval_object3=retrieval3,molecules=molecules)

else: # compare freechem, equchem of same object or freechem of two different objects
    figs.compare_retrievals(retrieval,retrieval2)  
    chems=[retrieval.chemistry,retrieval2.chemistry]
    if 'equchem' in chems or 'quequchem' in chems:
        molecules=['H2','He','H2O','H2(18)O','12CO','13CO','CH4','NH3']
        figs.VMR_plot(retrieval,retrieval_object2=retrieval2,molecules=molecules)
    else: # compare two freechem retrievals
        figs.ratios_cornerplot(retrieval,retrieval_object2=retrieval2)
        molecules=['13CO','HF','H2S','H2(18)O']
        retrieval.cross_correlation(molecules) # gets CCFs and ACFs of each molecule
        retrieval2.cross_correlation(molecules)
        figs.compare_two_CCFs(retrieval,retrieval2,molecules)

