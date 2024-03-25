from target import *
from pRT_model import *
from likelihood import *
from parameters import *
print('starting retrieval')

M0355 = Target('2M0355')
ra=M0355.ra
dec=M0355.dec
JD=M0355.JD
data_wave,data_flux,data_err=M0355.load_spectrum()

constant_params = {#'log_g': 4.65,
                    #'T1': 2500,
                    #'T2': 1800,
                    #'T3': 1400,
                    #'T4': 1100,
                    #'T5': 900,
                    #'T6': 800,
                    #'T7': 700,
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
free_params = {'vsini':([1.0,20.0],r'$v \sin(i)$ [km/s]'),
               'rv':([-30.0,30.0],r'RV [km/s]'),
               'log_H2O':([-12,-1],r'H$_2$O'),
               'log_12CO':([-12,-1],r'$^{12}$CO'),
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
               'T1':([0,5000],r'T1'),
               'T2':([0,5000],r'T2'),
               'T3':([0,5000],r'T3'),
               'T4':([0,5000],r'T4'),
               'T5':([0,5000],r'T5'),
               'T6':([0,5000],r'T6'),
               'T7':([0,5000],r'T7'),
               'log_g':([3,7],r'log$g$'),
              }

parameters = Parameters(free_params, constant_params)
cube = np.random.rand(parameters.n_params)
parameters(cube)
params=parameters.params

output='2M0355_test1'
retrieval=Retrieval(target=M0355,parameters=parameters,output_name=output)
retrieval.PMN_run(N_live_points=200,evidence_tolerance=0.5)
retrieval.PMN_analyse()
retrieval.cornerplot()
final_params=retrieval.get_final_parameters()
bestfit_model=retrieval.get_bestfit_model(plot_spectrum=True,plot_pt=True)