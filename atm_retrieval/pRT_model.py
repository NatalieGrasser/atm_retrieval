import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from PyAstronomy.pyasl import fastRotBroad
# SWITCH BROADENING TO https://github.com/Adolfo1519/RotBroadInt for large wavelength range and fast rotators
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import pathlib
import gc
from cloud_cond import simple_cdf_MgSiO3,return_XMgSiO3
from utils import *
import warnings
import re
from scipy.linalg import LinAlgWarning
from scipy.integrate import simps
#from scipy.constants import sigma, h, c, k as sc.sigma, sc.h, sc.c, sc.k
import scipy.constants as sc
from petitRADTRANS.physics import guillot_global
warnings.filterwarnings(action='ignore', category=LinAlgWarning) # occasional

import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
    path_tables = '/net/lem/data2/regt/fastchem_tables'
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    path_tables = '/home/natalie/fastchem_tables'

class pRT_spectrum:

    gc_n=0

    def __init__(self,
                 retr_obj, # retrieval object
                 contribution=False, # only for plotting atmosphere.contr_em
                 interpolate=True, # interpolate onto data wavlength grid
                 PT=None, # option to set a fixed PT profile to use
                 add_species=None, # get equ abund of species not included
                 leave_out=[]):
        
        inherit_attributes = ['primary_label','data_wave','instrument','species_pRT','name',
                              'chemistry','atmosphere_objects','n_atm_layers','species_info',
                              'pressure','PT_type','cloud_mode','spectral_resolution',
                              'mask_isfinite','data_flux','use_partial_pressure','species_names']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(retr_obj, attr, None))

        if hasattr(retr_obj, 'Teff_ref'):
            self.Teff_ref = retr_obj.Teff_ref

        if self.instrument=='CRIRES':
            self.lbl_opacity_sampling = retr_obj.lbl_opacity_sampling
            self.n_orders=retr_obj.n_orders
            self.n_dets=retr_obj.n_dets
            self.n_pixels = retr_obj.n_pixels

        if self.primary_label==False:
            self.primary_wave=retr_obj.primary_wave
            self.primary_flux=retr_obj.primary_flux
            self.data_flux = retr_obj.data_flux
            self.data_err = retr_obj.data_err

        self.params=retr_obj.parameters.params
        self.skewed_p_nodes = self.params['skewed_p_nodes'] if self.params['skewed_p_nodes']!=False else 1.
        self.interpolate=interpolate
        self.add_species = add_species
        if PT is None and self.params['fix_PT']==False:
            self.temperature = self.make_pt() #P-T profile
        elif PT is not None and self.params['fix_PT']==False:
            print('Using inputted PT')
            self.pressure = PT[0]
            self.temperature = PT[1]
        elif PT is None and self.params['fix_PT']==True and retr_obj.target.name in ['Sorg1X','Sorg20X']:
            psg_temperature = PSG_input(retr_obj.target.name).temperature
            psg_pressure = PSG_input(retr_obj.target.name).pressure
            self.temperature = np.interp(self.pressure, psg_pressure, psg_temperature)
        self.vbary = retr_obj.target.vbary
        self.gravity = 10**self.params['log_g']
            
        self.give_absorption_opacity=None
        self.int_opa_cloud = np.zeros_like(self.pressure)
        if self.instrument=='LIFE' and 'phot_fraction' in retr_obj.parameters.params:
            self.contribution=True
        else:
            self.contribution=contribution
        self.leave_out = leave_out if isinstance(leave_out, list) else [leave_out]

        # add_cloud_scat_as_abs, sigma_lnorm, fsed, Kzz only relevant for physical clouds (e.g. MgSiO3)
        self.sigma_lnorm=None
        self.Kzz=None
        self.fsed=None 
        self.add_cloud_scat_as_abs=False
        self.unphysical_params = False # flag when problems
        self.P_tot = 0.0
        self.P_surf = max(self.pressure)
        self.Teff_model = 0.0

        if 'log_k_rk' in self.params:
            wl_mid = np.median(self.data_wave)
            self.rk_func = lambda x: 10**self.params['log_k_rk']*np.array(x-wl_mid) + self.params['d_rk']

        if 'T_disk' in self.params:
            wl_mid = np.median(self.data_wave)
            #self.rk_func = lambda x: 10**self.params['log_k_rk']*np.array(x-wl_mid) + self.params['d_rk']
            self.BB_0 = planck_lambda_um(self.params['T_disk'],wl_mid*1e-3)
            self.rk_func = lambda x: self.params['phi_disk']*planck_lambda_um(self.params['T_disk'],x*1e-3)/self.BB_0

        if self.params['fix_all_except_PT']==True and retr_obj.target.name in ['Sorg1X','Sorg20X']:
            tab = PSG_input(retr_obj.target.name).table
            psg_pressure = PSG_input(retr_obj.target.name).pressure
            all_species = self.species_names.copy()
            all_species.extend(['H2','He'])
            self.VMR_dict = {}
            for species_i in all_species:
                vmr = tab[species_i].values # 100 layers
                self.VMR_dict[species_i] = np.interp(self.pressure, psg_pressure, vmr) # 50 layers
            self.mass_fractions = self.VMR_to_MF(self.VMR_dict)
            self.MMW = self.mass_fractions['MMW']
            self.FeH = 1.
            self.CO = 1. # just to avoid errors

        elif self.chemistry=='freechem': # use free chemistry with defined VMRs
            if self.use_partial_pressure:
                self.mass_fractions, self.CO, self.FeH = self.free_chemistry_use_partial_pressure(self.species_pRT,self.params)
                self.MMW = self.mass_fractions['MMW']
                self.VMR_dict = self.get_VMR_dict(self.mass_fractions)
            else:
                self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
        elif self.chemistry=='varchem':   
            if self.use_partial_pressure:
                self.mass_fractions, self.CO, self.FeH = self.var_chemistry_use_partial_pressure(self.species_pRT,self.params)
            else:
                self.mass_fractions, self.CO, self.FeH = self.var_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
            self.VMR_dict = self.get_VMR_dict(self.mass_fractions)

        elif self.chemistry in ['equchem','quequchem','flexequ']: # use equilibium chemistry
            self.species_hill = retr_obj.species_hill
            self.mass_fractions = self.equ_chemistry(self.species_pRT,self.params)
            # update mass_fractions with isotopolog ratios
            if any(key in self.params for key in ['13CO','C17O','C18O','H2(18)O','log_C12_13_ratio','log_O16_18_ratio','log_H2O16_18_ratio','log_O16_17_ratio']):
                self.mass_fractions = self.get_isotope_mass_fractions(self.species_names,self.species_pRT,self.mass_fractions,self.params) 
            self.MMW = self.mass_fractions['MMW']
            # get new VMR dict, updated with isotopologs
            self.VMR_dict = self.get_VMR_dict(self.mass_fractions)

            # keeps crashing for equchem??
            pRT_spectrum.gc_n+=1
            if pRT_spectrum.gc_n>20: # make it more efficient by not running it every time
                gc.collect()
                pRT_spectrum.gc_n=0  

    def get_VMR_dict(self,mass_fractions):
        VMR_dict={}
        MMW=self.MMW
        for pRT_name in mass_fractions.keys():
            if pRT_name!='MMW':
                row = self.species_info[self.species_info["pRT_name"] == pRT_name]
                mass = row["mass"].values[0]
                name = row.index[0]
                VMR_dict[name]=mass_fractions[pRT_name]*MMW/mass
        return VMR_dict
    
    def read_species_info(self,species,info_key):
        if info_key == 'pRT_name':
            return self.species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return self.species_info.loc[species,'Hill_notation']
        if info_key == 'mass':
            return self.species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(self.species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return self.species_info.loc[species,info_key]
        if info_key == 'c' or info_key == 'color':
            return self.species_info.loc[species,'color']
        if info_key == 'label':
            return self.species_info.loc[species,'mathtext_name']
    
    def get_isotope_mass_fractions(self,species_names,species_pRT,mass_fractions,params):

        mass_ratio_13CO_12CO = self.read_species_info('13CO','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C18O_C16O = self.read_species_info('C18O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C17O_C16O = self.read_species_info('C17O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_H218O_H2O = self.read_species_info('H2(18)O','mass')/self.read_species_info('H2O','mass')
        self.C13_12_ratio = 10**(-params.get('log_C12_13_ratio',15))
        self.O18_16_ratio = 10**(-params.get('log_O16_18_ratio',15))
        self.H2O18_16_ratio = 10**(-params.get('log_H2O16_18_ratio',15))
        self.O17_16_ratio = 10**(-params.get('log_O16_17_ratio',15))

        for species_i,species_pRT_i in zip(species_names,species_pRT):
            if (species_pRT_i in ['CO_main_iso','CO_high']): # 12CO mass fraction
                CO_linelist = species_pRT_i
                mass_fractions[species_pRT_i]=(1-self.C13_12_ratio*mass_ratio_13CO_12CO
                                            -self.O18_16_ratio*mass_ratio_C18O_C16O
                                            -self.O17_16_ratio*mass_ratio_C17O_C16O)*mass_fractions[CO_linelist]
                continue
            if (species_pRT_i in ['CO_36','CO_36_high']) and (species_i not in self.leave_out): # 13CO mass fraction
                mass_fractions[species_pRT_i]=self.C13_12_ratio*mass_ratio_13CO_12CO*mass_fractions[CO_linelist]
                continue
            if (species_pRT_i in ['CO_28','CO_28_high_Sam']) and (species_i not in self.leave_out): # C18O mass fraction
                mass_fractions[species_pRT_i]=self.O18_16_ratio*mass_ratio_C18O_C16O*mass_fractions[CO_linelist]
                continue
            if (species_pRT_i in ['CO_27','CO_27_high_Sam']) and (species_i not in self.leave_out): # C17O mass fraction
                mass_fractions[species_pRT_i]=self.O17_16_ratio*mass_ratio_C17O_C16O*mass_fractions[CO_linelist]
                continue
            if (species_pRT_i in ['H2O_main_iso','H2O_pokazatel_main_iso']): # H2O mass fraction
                H2O_linelist=species_pRT_i
                mass_fractions[species_pRT_i]=(1-self.H2O18_16_ratio*mass_ratio_H218O_H2O)*mass_fractions[H2O_linelist]
                continue
            if (species_pRT_i=='H2O_181_HotWat78') and (species_i not in self.leave_out): # H2_18O mass fraction
                mass_fractions[species_pRT_i]=self.H2O18_16_ratio*mass_ratio_H218O_H2O*mass_fractions[H2O_linelist]
                continue
            
        return mass_fractions
    
    def VMR_to_MF(self, VMRs):
        MMW = 0.
        for species_i, VMR_i in VMRs.items():
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i

        # Convert to mass-fractions using mass-ratio
        self.mass_fractions = {'MMW': MMW * np.ones(self.n_atm_layers)}
        for species_i, VMR_i in VMRs.items():
            species_pRT_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            mf = VMR_i * mass_i / MMW
            self.mass_fractions[species_pRT_i] = mf
        return self.mass_fractions

    def equ_chemistry(self, species_pRT, params):

        species_pRT.extend(s for s in ['H2','He'] if s not in species_pRT)
        self.species_hill.extend(s for s in ['H2','He'] if s not in self.species_hill)

        if 'H-' in species_pRT:  # required for calculation
            species_pRT.extend(s for s in ['e-','H'] if s not in species_pRT)
            self.species_hill.extend(s for s in ['e-','H'] if s not in self.species_hill)
        
        if self.add_species!=None:
            species_pRT.extend((self.species_info.loc[self.add_species,'pRT_name']))
            self.species_hill.extend((self.species_info.loc[self.add_species,'Hill_notation']))

        def load_interp_tables():
            import h5py, pathlib
            def load_hdf5(file, key):
                with h5py.File(f'{path_tables}/{file}', 'r') as f:
                    return f[key][...]

            # Load the interpolation grid (ignore N/O)
            self.P_grid  = load_hdf5('grid.hdf5', 'P')
            self.T_grid  = load_hdf5('grid.hdf5', 'T')
            self.CO_grid = load_hdf5('grid.hdf5', 'C/O')
            self.FeH_grid = load_hdf5('grid.hdf5', 'Fe/H')
            points = (self.P_grid, self.T_grid, self.CO_grid, self.FeH_grid)

            self.interp_tables = {}
            for species_i, hill_i in zip([*species_pRT, 'MMW'], [*self.species_hill, 'MMW']):
                key = 'MMW' if species_i == 'MMW' else 'log_VMR'
                if species_i in ['e-', 'H']:
                    hill_i = 'e-' if species_i == 'e-' else 'H'
                equ_table = pathlib.Path(f'{path_tables}/{hill_i}.hdf5')
                if equ_table.exists():
                    arr = load_hdf5(f'{hill_i}.hdf5', key=key)  # Load equchem abundance tables
                self.interp_tables[species_i] = RegularGridInterpolator(
                    values=arr[:,:,:,0,:], points=points, method='linear'
                )

        def get_VMRs(ParamTable):
            self.VMRs = {}

            def apply_bounds(val, grid):
                val = np.array(val)
                val[val > grid.max()] = grid.max()
                val[val < grid.min()] = grid.min()
                return val

            # Update the parameters
            self.CO  = ParamTable.get('C/O')
            self.FeH = ParamTable.get('Fe/H')

            # Apply the bounds of the grid
            P = apply_bounds(self.pressure.copy(), grid=self.P_grid)
            T = self.temperature.copy()
            CO  = apply_bounds(np.array([self.CO]).copy(), grid=self.CO_grid)[0]
            FeH = apply_bounds(np.array([self.FeH]).copy(), grid=self.FeH_grid)[0]

            T_max = self.T_grid.max()

            # Interpolate abundances
            for pRT_name_i, interp_func_i in self.interp_tables.items():

                # Clip T for interpolation (avoid out-of-bounds)
                T_clip = np.clip(T, self.T_grid.min(), self.T_grid.max())
                arr_i = interp_func_i(xi=(P, T_clip, CO, FeH))

                # hold VMR constant above 6000K, limit of equchem tables
                # Find the last valid layer below T_max
                valid_mask = T <= T_max
                if np.any(valid_mask):
                    last_valid_idx = np.where(valid_mask)[0][-1]
                    arr_i[~valid_mask] = arr_i[last_valid_idx]

                if pRT_name_i != 'MMW':
                    species_i = self.species_info[self.species_info["pRT_name"] == pRT_name_i].index[0]

                    if self.chemistry == 'flexequ' and species_i not in ['13CO','C17O','C18O','H2(18)O']:
                        vmr = (10**arr_i) * (10**params[f'log_a_{species_i}'])
                        self.VMRs[species_i] = np.clip(vmr, a_min=None, a_max=0.1)
                    else:
                        self.VMRs[species_i] = 10**arr_i  # log10(VMR)

                    if species_i in self.leave_out:
                        self.VMRs[species_i].fill(0)
                else:
                    self.MMW = arr_i.copy()  # Mean molecular weight
            return self.VMRs

        load_interp_tables()
        self.VMRs = get_VMRs(params)
        self.mass_fractions = self.VMR_to_MF(self.VMRs)

        if self.chemistry == 'quequchem':
            for species in self.mass_fractions.keys():
                if any(sub in species for sub in ['H2O_','CO_','CH4_']) and 'log_Pqu_CO_CH4' in self.params:
                    Pqu = 10**self.params['log_Pqu_CO_CH4']
                    idx = find_nearest(self.pressure, Pqu)
                    quenched_fraction = self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx] = quenched_fraction
                elif any(sub in species for sub in ['H2O_','OH_','O_']) and 'log_Pqu_H2O_OH_O' in self.params:
                    Pqu = 10**self.params['log_Pqu_H2O_OH_O']
                    idx = find_nearest(self.pressure, Pqu)
                    quenched_fraction = self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx] = quenched_fraction
                elif 'NH3_' in species and 'log_Pqu_NH3' in self.params:
                    Pqu = 10**self.params['log_Pqu_NH3']
                    idx = find_nearest(self.pressure, Pqu)
                    quenched_fraction = self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx] = quenched_fraction
                elif 'HCN_' in species and 'log_Pqu_HCN' in self.params:
                    Pqu = 10**self.params['log_Pqu_HCN']
                    idx = find_nearest(self.pressure, Pqu)
                    quenched_fraction = self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx] = quenched_fraction

        return self.mass_fractions
    
    def free_chemistry(self,species_pRT,params):
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0
        if 'log_e-' in self.params:
            species_pRT.append('e-') if 'e-' not in species_pRT else None

        for species_i in self.species_info.index:
            species_pRT_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if species_pRT_i in species_pRT:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[species_pRT_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species

        if VMR_wo_H2.any() > 1:
            print('\nVMR_wo_H2 > 1. Other species are too abundant!',VMR_wo_H2)

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for species_pRT_i in mass_fractions.keys():
            mass_fractions[species_pRT_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O if np.sum(O)!=0 else np.inf
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar if (C/H).any()!=0.0 else -np.inf*np.ones((len(C)))
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        return mass_fractions, CO, FeH

    def var_chemistry(self,line_species,params): # vary with pressure

        CO_list=[]
        FeH_list=[]
        VMR_He = 0.15
        VMRs_list=[]

        # check how many pressure knots
        n_knots = sum(1 for key in params if re.fullmatch(r'log_H2O_\d+', key))

        for knot in range(n_knots): # points where to retrieve abundances

            VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
            VMRs={}
            C, O, H = 0, 0, 0

            for species_i in self.species_info.index:
                line_species_i = self.read_species_info(species_i,'pRT_name')
                mass_i = self.read_species_info(species_i, 'mass')
                COH_i  = self.read_species_info(species_i, 'COH')

                if species_i in ['H2', 'He']:
                    continue
                if line_species_i in line_species:
                    if f'log_{species_i}_{knot}' in params:
                        VMR_i = 10**(params[f'log_{species_i}_{knot}'])
                    else:
                        VMR_i = 10**(params[f'log_{species_i}']) # vertically constant for some species
                    # Convert VMR to mass fraction using molecular mass number
                    VMRs[line_species_i] = VMR_i
                    VMR_wo_H2 += VMR_i

                    # Record C, O, and H bearing species for C/O and metallicity
                    C += COH_i[0] * VMR_i
                    O += COH_i[1] * VMR_i
                    H += COH_i[2] * VMR_i

            # Add the H2 and He abundances
            VMRs['He'] = VMR_He
            H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
            self.VMR_wo_H2=VMR_wo_H2

            CO = C/O
            log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
            FeH = np.log10(C/H)-log_CH_solar
            CO_list.append(CO)
            FeH_list.append(FeH)
            VMRs_list.append(VMRs)

        VMRs_interp = {}
        mass_fractions_interp = {}
        line_species.append('He')

        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            if line_species_i in line_species and line_species_i!='H2':
                vmrs_var = []
                for kn in range(n_knots):
                    vmrs_var.append(VMRs_list[kn][line_species_i])
                log_P_knots = generate_skewed_p_nodes(n_nodes=n_knots, skew=self.skewed_p_nodes)
                #else:
                    #log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=n_knots)

                # use linear interpolation to avoid going into negative values cubic spline did that)
                log_vmrs=np.interp(np.log10(self.pressure), log_P_knots, np.log10(vmrs_var)) # interpolate for all layers

                VMRs_interp[line_species_i] = 10**log_vmrs #np.interp(np.log10(self.pressure), log_P_knots, mass_fracs) # interpolate for all layers
                mass_fractions_interp[line_species_i]=mass_i*VMRs_interp[line_species_i]

        vmr_layers = np.empty(self.n_atm_layers) # vmr of layers
        for l in range(self.n_atm_layers):
            vmr=0
            for key in VMRs_interp.keys():
                vmr+=VMRs_interp[key][l]
            vmr_layers[l]=vmr

        self.vmr_layers=vmr_layers
        vmr_H2=np.empty(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            vmr_H2[l]=1-vmr_layers[l]
            if vmr_H2[l]<0:
                print('Invalid VMR',vmr_H2[l])
                #print(mass_fractions_interp)
                self.VMR_wo_H2=1.1
                exit_mf={}
                for key in VMRs_interp.keys():
                    exit_mf[key]=np.ones(self.n_atm_layers)*1e-12
                return exit_mf,1,1
        VMRs_interp['H2']=vmr_H2
        mass_fractions_interp['H2']=self.read_species_info('H2','mass')*VMRs_interp['H2']

        mmw_layers=np.empty(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            MMW = 0 # Compute the mean molecular weight from all species for each layer
            for line_species_i in mass_fractions_interp.keys():
                #print(line_species_i, mass_fractions_interp[line_species_i][l])
                MMW += mass_fractions_interp[line_species_i][l]
            mmw_layers[l] = MMW
        mass_fractions_interp['MMW'] = mmw_layers # pRT requires MMW in mass fractions dictionary

        for line_species_i in mass_fractions_interp.keys():
            if line_species_i=='MMW':
                continue
            mass_fractions_interp[line_species_i] /= mass_fractions_interp['MMW'] # Turn the molecular masses into mass fractions
                       
        CO = np.nanmean(CO_list)
        FeH = np.nanmean(FeH_list)
        self.VMRs=VMRs_interp
        
        return mass_fractions_interp, CO, FeH

    def free_chemistry_use_partial_pressure(self, species_pRT, params):
        """
        Free chemistry using retrieved log partial pressures [bar] for ALL species, 
        including H2 and He. The total pressure in each layer is reconstructed from the sum.
        """

        mass_fractions = {}
        C, O, H = 0, 0, 0
        species_pRT.append('He')
        species_pRT.append('H2')

        # Retrieve partial pressures for all species in each layer
        P_partials = {}

        for species_i in self.species_info.index:
            species_pRT_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i = self.read_species_info(species_i, 'COH')

            if species_pRT_i in species_pRT:
                # Retrieve one parameter per species (constant with altitude)
                logP_i = params[f'log_p_{species_i}']
                P_partials[species_i] = 10**logP_i #* np.ones(self.n_atm_layers)  # [bar]

        # Total pressure profile reconstructed as sum of retrieved partial pressures
        P_tot = 0.0
        for P_i in P_partials.values():
            P_tot += P_i
        if P_tot > max(self.pressure):
            self.unphysical_params = True

        self.P_tot = P_tot

        # Convert partial pressures to VMR and compute mass fractions
        VMR_tot = 0.0
        for species_i, P_i in P_partials.items():
            VMR_i = P_i / P_tot
            VMR_tot += VMR_i
            species_pRT_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i = self.read_species_info(species_i, 'COH')

            mass_fractions[species_pRT_i] = mass_i * VMR_i
            C += COH_i[0] * VMR_i
            O += COH_i[1] * VMR_i
            H += COH_i[2] * VMR_i

        # Compute MMW
        MMW = sum(mass_fractions.values()) * np.ones(self.n_atm_layers)

        # Normalize mass fractions
        for species_pRT_i in mass_fractions.keys():
            mass_fractions[species_pRT_i] /= MMW
        mass_fractions['MMW'] = MMW

        # Element ratios
        CO = np.nanmean(C / O) if np.sum(O) != 0 else np.inf
        log_CH_solar = 8.46 - 12  # Asplund et al. (2021)
        FeH = np.nanmean(np.log10(C / H) - log_CH_solar) if np.any(H) else -np.inf

        return mass_fractions, CO, FeH

    def var_chemistry_use_partial_pressure(self, line_species, params):
        CO_list = []
        FeH_list = []
        line_species.append('He')
        line_species.append('H2')
        
        n_knots = sum(1 for key in params if re.fullmatch(r'log_p_H2O_\d+', key))  # knots for vertical retrieval
        
        use_partial_pressure_list = []
        for knot in range(n_knots):
            use_partial_pressure_sum = 0
            use_partial_pressure_knot = {}
            C, O, H = 0, 0, 0

            for species_i in self.species_info.index:
                line_species_i = self.read_species_info(species_i, 'pRT_name')
                mass_i = self.read_species_info(species_i, 'mass')
                COH_i = self.read_species_info(species_i, 'COH')

                # Retrieve partial pressure at knot for all species
                key = f'log_p_{species_i}_{knot}'
                if key in params:
                    p_i = 10 ** params[key]
                else:
                    # fallback to vertically constant partial pressure if no knot key
                    const_key = f'log_p_{species_i}'
                    if const_key in params:
                        p_i = 10 ** params[const_key]
                    else:
                        p_i = 0  # or some very low floor if desired

                use_partial_pressure_knot[species_i] = p_i  # <-- use species_i here for VMR keys
                use_partial_pressure_sum += p_i

                # Calculate C, O, H contributions for metallicities
                C += COH_i[0] * p_i
                O += COH_i[1] * p_i
                H += COH_i[2] * p_i

            # Normalize C/O and Fe/H by total partial pressure at this knot (i.e. total pressure)
            CO = C / O if O != 0 else np.inf
            log_CH_solar = 8.46 - 12  # Asplund et al. (2021)
            FeH = np.log10(C / H) - log_CH_solar if H != 0 else -np.inf

            CO_list.append(CO)
            FeH_list.append(FeH)
            use_partial_pressure_list.append(use_partial_pressure_knot)

        # Interpolate partial pressures over layers:
        use_partial_pressure_interp = {}
        mass_fractions_interp = {}

        # Append He to line_species if not present to interpolate
        if 'He' not in line_species:
            line_species.append('He')

        # Generate knot locations in log pressure space:
        log_P_knots = generate_skewed_p_nodes(n_nodes=n_knots, skew=self.skewed_p_nodes)

        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            if line_species_i in line_species:
                p_knots = []
                for kn in range(n_knots):
                    # Here use_partial_pressure_list uses species_i keys
                    p_knots.append(use_partial_pressure_list[kn].get(species_i, 0))
                
                # interpolate log partial pressures over atmospheric layers
                # add a tiny floor to avoid log(0)
                p_knots_safe = np.array(p_knots)
                p_knots_safe[p_knots_safe == 0] = 1e-30
                log_p_knots = np.log10(p_knots_safe)
                log_pressure_layers = np.log10(self.pressure)

                log_p_interp = np.interp(log_pressure_layers, log_P_knots, log_p_knots)
                use_partial_pressure_interp[species_i] = 10 ** log_p_interp  # <-- key species_i for VMRs

        # Calculate total pressure per layer by summing partial pressures
        use_partial_pressure_array = np.vstack([use_partial_pressure_interp[spec] for spec in use_partial_pressure_interp])
        P_tot_layers = np.sum(use_partial_pressure_array, axis=0)

        # Calculate VMRs = p_i / P_tot per layer (keys species_i)
        VMRs_interp = {}
        for spec in use_partial_pressure_interp:
            VMRs_interp[spec] = use_partial_pressure_interp[spec] / P_tot_layers

        # Calculate mass fractions from VMRs (keys line_species_i)
        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            if species_i in VMRs_interp:
                mass_i = self.read_species_info(species_i, 'mass')
                mass_fractions_interp[line_species_i] = mass_i * VMRs_interp[species_i]

        # Calculate MMW per layer
        mmw_layers = np.zeros(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            mmw_layers[l] = sum(mass_fractions_interp[spec][l] for spec in mass_fractions_interp)

        mass_fractions_interp['MMW'] = mmw_layers

        # Normalize mass fractions by MMW per layer
        for spec in mass_fractions_interp:
            if spec == 'MMW':
                continue
            mass_fractions_interp[spec] /= mmw_layers

        CO = np.nanmean(CO_list)
        FeH = np.nanmean(FeH_list)

        self.VMRs = VMRs_interp
        self.P_tot = np.median(P_tot_layers)  # optionally store total pressure per layer

        return mass_fractions_interp, CO, FeH

    def gray_cloud_opacity(self,wave_micron,pressure): # gray cloud = independent of wavelength
        P_base_gray = 10**(self.params['log_P_base_gray'])
        opa_base_gray = 10**(self.params['log_opa_base_gray'])
        if self.params.get('cloud_slope') is not None:
            opa_gray_cloud = np.zeros((len(wave_micron),len(pressure)))
            opa_gray_cloud[:,pressure>P_base_gray] = 0 # [bar] constant below cloud base
            opa_gray_cloud[:,pressure<=P_base_gray]=opa_base_gray*(pressure[pressure<=P_base_gray]/P_base_gray)**self.params['fsed_gray']
            opa_gray_cloud *= (wave_micron[:,None]/1)**self.params['cloud_slope']
        else:
            opa_gray_cloud = np.zeros(len(pressure)) # no need for wavelength dimension
            opa_gray_cloud[pressure>P_base_gray] = 0 # [bar] constant below cloud base
            opa_gray_cloud[pressure<=P_base_gray]=opa_base_gray*(pressure[pressure<=P_base_gray]/P_base_gray)**self.params['fsed_gray']     
        #pRT_spectrum.gc_n+=1
        #if pRT_spectrum.gc_n>20: # make it more efficient by not running it every time
            #gc.collect()
            #pRT_spectrum.gc_n=0      
        return opa_gray_cloud
    
    def make_spectrum(self,save_whole_contribution=False):
        
        self.norm_factor_A = None
        self.norm_factor_B = None
        summed_emconts =[]
        self.model_continuum = np.ones(self.data_wave.shape)

        if self.instrument=='CRIRES':
            data_shape = self.data_wave.shape
            orders_shape = (self.n_orders,self.n_dets*self.n_pixels)
            data_wave_orders = self.data_wave.reshape(orders_shape)
            
            if self.primary_label==False:
                orders_dets_shape = (self.n_orders,self.n_dets,self.n_pixels)
                self.data_flux = self.data_flux.reshape(orders_dets_shape)
                self.data_err = self.data_err.reshape(orders_dets_shape)
                self.primary_wave=self.primary_wave.reshape(orders_dets_shape)
                self.primary_flux=self.primary_flux.reshape(orders_dets_shape)
                self.secondary_flux=np.full(shape=orders_dets_shape,fill_value=np.nan)
                self.primary_broadened=np.full(shape=orders_dets_shape,fill_value=np.nan)
                self.continuum_flux=np.full(shape=(7,3*2048),fill_value=np.nan)

            spectrum_parts=[]
            waves_parts=[]

        if isinstance(self.atmosphere_objects, list)==False:
            self.atmosphere_objects = [self.atmosphere_objects]

        for part, atmosphere in enumerate(self.atmosphere_objects):

            if self.cloud_mode == 'MgSiO3':
                if self.chemistry=='freechem':
                    co=self.CO
                    feh=self.FeH
                if self.chemistry in ['equchem','quequchem','flexequ']:
                    feh=self.params['Fe/H']
                    co=self.params['C/O']
                P_base_MgSiO3 = simple_cdf_MgSiO3(self.pressure,self.temperature,feh,co,np.nanmean(self.MMW))
                above_clouds = (self.pressure<=P_base_MgSiO3) # mask pressure above cloud deck
                eq_MgSiO3 = return_XMgSiO3(feh, co)
                self.mass_fractions['MgSiO3(c)'] = np.zeros_like(self.temperature)
                condition=eq_MgSiO3*(self.pressure[above_clouds]/P_base_MgSiO3)**self.params['fsed']
                self.mass_fractions['MgSiO3(c)'][above_clouds]=condition
                self.sigma_lnorm=self.params['sigma_lnorm']
                self.Kzz = 10**self.params['log_Kzz']*np.ones_like(self.pressure) 
                self.fsed=self.params['fsed']
                self.add_cloud_scat_as_abs=True

            elif self.cloud_mode == 'gray': # Gray cloud opacity
                self.give_absorption_opacity=self.gray_cloud_opacity # fsed_gray only needed here, not in calc_flux

            atmosphere.setup_opa_structure(self.pressure)
            atmosphere.calc_flux(self.temperature,
                            self.mass_fractions,
                            self.gravity,
                            self.MMW,
                            Kzz=self.Kzz, # only for MgSiO3 clouds
                            fsed = self.fsed, # only for MgSiO3 clouds 
                            sigma_lnorm = self.sigma_lnorm, # only for MgSiO3 clouds
                            add_cloud_scat_as_abs=self.add_cloud_scat_as_abs, # only for MgSiO3 clouds
                            contribution =self.contribution,
                            give_absorption_opacity=self.give_absorption_opacity)

            wl_cm = const.c.to(u.km/u.s).value/atmosphere.freq/1e-5 # cm
            # [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
            flux = atmosphere.flux*const.c.to(u.km/u.s).value/(wl_cm**2) # convert from flux density to flux
            wl = wl_cm/1e-4 # microns

            if self.contribution==True: # emission contribution
                self.summed_emcont = np.nansum(atmosphere.contr_em,axis=1) # sum over all wavelengths
                summed_emconts.append(self.summed_emcont)
                if save_whole_contribution:
                    self.emission_contribution = atmosphere.contr_em
                    self.wave_um = wl

            if self.instrument=='CRIRES':
                # RV+bary shifting and rotational broadening
                waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
                wl_shifted= wl*(1.0+(self.params['rv']-self.vbary)/const.c.to('km/s').value)
                flux = np.interp(waves_even, wl_shifted, flux)
                flux = fastRotBroad(waves_even, flux, self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
                flux = self.instrumental_broadening(waves_even, flux, self.spectral_resolution)

            if self.instrument=='LIFE':

                if 'Teff_ref' in self.params:

                    # estimate T_eff from flux
                    wl_cm = (const.c / atmosphere.freq).cgs.value  # cm
                    wl_micron = wl_cm * 1e4
                    F_nu = atmosphere.flux  # erg/s/cm²/Hz
                    F_lambda = F_nu * const.c.cgs.value / wl_cm**2  # erg/s/cm²/cm
                    sort_idx = np.argsort(wl_cm)
                    wl_cm = wl_cm[sort_idx]
                    F_lambda = F_lambda[sort_idx]

                    # Integrate observed/retrieved F_lambda (erg/s/cm²/cm)
                    flux_int = simps(F_lambda, wl_cm)  # erg / s / cm²

                    # Convert to W/m²
                    flux_W_m2 = flux_int * 1e-7 * 1e4  # erg → J ; cm^-2 → m^-2

                    # Compute Teff from partial observed range (underestimated)
                    Teff_partial = (flux_W_m2 / sc.sigma)**0.25 
                    T_ref = self.Teff_ref  # Reference Teff from literature

                    # calc reference Planck curves for correction
                    wl_full_um = np.linspace(1.0, 100.0, 1000)  # Full reference range
                    B_full = planck_lambda_um(T_ref, wl_full_um)
                    B_model = planck_lambda_um(T_ref, wl_micron)
                    B_int_full = simps(B_full, wl_full_um)
                    B_int_partial = simps(B_model, wl_micron)
                    flux_fraction = B_int_partial / B_int_full
                    Teff_corrected = Teff_partial / flux_fraction**0.25
                    self.Teff_model = Teff_corrected
   
                wl_um,flux_observed = self.pRT_to_photon_flux(atmosphere)
                flux_observed = self.instrumental_broadening(wl_um,flux_observed,self.spectral_resolution)
                flux_model = np.interp(self.data_wave.flatten(), wl_um, flux_observed)
                flux_model = flux_model.reshape(self.data_flux.shape)
                
                if np.nanmax(flux_model) in [0, np.nan, np.inf] or len(flux_model)==0 or self.unphysical_params==True:
                    #print('\nInvalid flux, max=',np.nanmax(flux_model))
                    return np.ones_like(self.data_flux)*np.nanmedian(self.data_flux)
                else:
                    return flux_model #flux/np.nanmax(flux)

            # Interpolate/rebin onto the data's wavelength grid
            # should not be done when making spectrum for cross-corr, or wavelength padding will be cut off
            if self.interpolate==True:
                ref_wave = data_wave_orders[part]# [nm]
                flux = np.interp(ref_wave, waves_even*1e3, flux) # pRT wavelengths from microns to nm

            if self.primary_label==False and self.interpolate==True: # should have same wavelengths
                order=part
                
                flux = flux.reshape(self.n_dets,self.n_pixels)
                for det in range(3):

                    nonans = np.isfinite(self.primary_flux[order][det]) & np.isfinite(self.data_flux[order][det]) & np.isfinite(self.data_err[order][det])
                    wl_det = self.data_wave.reshape(self.n_orders,self.n_dets,self.n_pixels)[order,det]
                    
                    if ('log_k_rk' in self.params) or ('T_disk' in self.params): # veiling
                        flB = np.copy(flux[det])
                        rk = self.rk_func(wl_det)
                        flux[det] = ((flB+rk*np.nanmedian(flB))/(1+rk))               
                    if np.sum(nonans)==0:
                        continue
                    
                    star = self.primary_flux[order,det][nonans] # primary (star)
                    star = self.instrumental_broadening(wl_det,star,fwhm=self.params['star_fwhm']) # broaden starlight

                    #normalize by const to keep slope
                    if self.norm_factor_A==None and self.norm_factor_B==None:
                        self.norm_factor_A = np.nanmedian(star)
                        self.norm_factor_B = np.nanmedian(flux[det])
                    star/=self.norm_factor_A
                    flux[det]/=self.norm_factor_B

                    if 'log_phi_k' in self.params: # as linear func
                        wl_mid = np.median(self.data_wave) # intercept defined here
                        phi_k = 10**self.params['log_phi_k']
                        phi_d = self.params['phi_d']
                        phi_secondary = np.array(phi_k*(wl_det-wl_mid) + phi_d)
                    elif 'phi_secondary' in self.params:
                        phi_secondary = self.params['phi_secondary']*np.ones_like(wl_det)
                    phi_primary = np.ones_like(phi_secondary)-phi_secondary

                    self.secondary_flux[order,det]=phi_secondary*np.copy(flux[det])
                    self.primary_broadened[order,det][nonans] = phi_primary[nonans]*star
                    total_flux = self.secondary_flux[order,det] + self.primary_broadened[order,det]                
                    flux[det]=total_flux

            spectrum_parts.append(flux)
            waves_parts.append(waves_even*1e3) # from um to nm

            if self.contribution==True:
                self.summed_emcont = np.nanmean(summed_emconts,axis=0)  

        if self.interpolate==False:
            get_median=np.array([])
            for order in range(7): # append value by value because not all the same size
                get_median=np.append(get_median,spectrum_parts[order]) 
            spectrum_parts=np.array(spectrum_parts,dtype=object)
            spectrum_parts/=np.nanmedian(get_median) # orders not same size, np.median didn't work otherwise
            if self.name in ['ROXs12A','ROXs12_onlyB','test_ROXs12B']: # normalized differently
                for order in range(7):
                    fac = len(spectrum_parts[order])/2048
                    spectrum_parts[order] = fft_remove_continuum(spectrum_parts[order],lower_cutoff=13*fac) # 3 dets+overhead
            return spectrum_parts, waves_parts
        else:
            spectrum_parts=np.array(spectrum_parts)
            spectrum_parts = spectrum_parts.reshape(data_shape)
            if self.primary_label==False:
                data_flux = self.data_flux.reshape(data_shape)
                self.continuum_flux = self.continuum_flux.reshape(data_shape)
                for i,part in enumerate(spectrum_parts):
                    if np.isnan(data_flux[i]).all():
                        continue
                    spectrum_parts[i] = fft_remove_continuum(spectrum_parts[i])
                return spectrum_parts
            else: # normalize in same way as data spectrum
                
                if self.name in ['ROXs12A','ROXs12_onlyB','test_ROXs12B']: # normalized differently
                    for i,part in enumerate(spectrum_parts):
                        if ('log_k_rk' in self.params) or ('T_disk' in self.params):
                            spectrum_parts[i] /=np.nanpercentile(spectrum_parts[i],97)
                            rk = self.rk_func(self.data_wave[i])
                            sp = spectrum_parts[i]
                            spectrum_parts[i] = (sp+rk*np.nanmedian(sp))/(1+rk)
                        spectrum_parts[i] /= np.nanmedian(spectrum_parts[i])
                        spectrum_parts[i][~self.mask_isfinite[i]] = np.nan
                        spec, continuum = fft_remove_continuum(spectrum_parts[i],orig_method=True, return_continuum=True)
                        spectrum_parts[i] =spec 
                        self.model_continuum[i] = continuum
                else:
                    spectrum_parts/=np.nanmedian(spectrum_parts) 
                
                return spectrum_parts
            
    def make_pt(self,**kwargs): 

        if self.PT_type=='PTknot': # retrieve temperature knots
            t_keys = [key for key in self.params.keys() if re.fullmatch(r"T\d+", key)] 
            t_keys = sorted(t_keys, key=lambda x: int(x[1:]))[::-1] # start at top of atmosphere, T0 last
            self.T_knots = []
            for key in t_keys:
                self.T_knots.append(self.params[key])
            self.T_knots = np.array(self.T_knots)
            self.log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=len(self.T_knots))
            sort = np.argsort(self.log_P_knots)
            self.temperature = CubicSpline(self.log_P_knots[sort],self.T_knots[sort])(np.log10(self.pressure))
        
        if self.PT_type in ['PTgrad','PTgradvar']:
            # check how many pressure knots
            n_grad = sum(1 for key in self.params if re.fullmatch(r'dlnT_dlnP_\d+', key))

            if self.PT_type=='PTgradvar': # varying pressure points
                self.log_P_knots = np.empty(n_grad)
                log_max_p = np.log10(np.max(self.pressure))
                log_min_p = np.log10(np.min(self.pressure))
                self.log_P_knots[0] = log_max_p
                self.log_P_knots[-1] = log_min_p
                n_array = np.linspace(1,n_grad,n_grad,dtype=int)[:-2]
                central_n = int(np.median(n_array))
                self.log_P_knots[central_n] = self.params['log_P_RCB']
                n_iter = int((len(n_array)-1)/2)
                for n in range(n_iter):
                    n+=1
                    p_below = np.log10(10**self.params['log_P_RCB']/10**self.params['log_delta_P'])
                    p_above = np.log10(10**self.params['log_P_RCB']*10**self.params['log_delta_P'])
                    self.log_P_knots[central_n-n] = max(np.log10(10**(log_min_p)*1.1), p_below)
                    self.log_P_knots[central_n+n] = min(np.log10(10**(log_max_p)*0.9), p_above)
                 
                if n_grad%2==0: # even number of PT points
                    p_below = np.log10(10**self.params['log_P_RCB']-2*10**(self.params['log_delta_P']))
                    n+=1 # n from previous loop
                    self.log_P_knots[central_n-n] = max(log_min_p, p_below)

            else: # use uniformly spaced PT-knots
                self.log_P_knots = np.linspace(np.log10(np.max(self.pressure)),
                                            np.log10(np.min(self.pressure)),
                                            num=n_grad)

            if 'Sorg' in self.name: # denser at bottom
                self.log_P_knots = generate_skewed_p_nodes(n_nodes=n_grad, skew=self.skewed_p_nodes)

            if 'dlnT_dlnP_knots' not in kwargs:
                self.dlnT_dlnP_knots=[]
                for i in range(n_grad):
                    self.dlnT_dlnP_knots.append(self.params[f'dlnT_dlnP_{i}'])
            elif 'dlnT_dlnP_knots' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                self.dlnT_dlnP_knots=kwargs.get('dlnT_dlnP_knots')

            # interpolate over dlnT/dlnP gradients
            interp_func = interp1d(self.log_P_knots,self.dlnT_dlnP_knots,kind='quadratic') # for the other 50 atm layers
            dlnT_dlnP = interp_func(np.log10(self.pressure)) # T0 & P0 at beginning, start at bottom of atm

            if 'T_base' not in kwargs:
                T_base = self.params['T0'] # T0 is free param, at bottom of atmosphere
            elif 'T_base' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                T_base=kwargs.get('T_base')

            ln_P = np.log(self.pressure)[::-1]
            temperature = [T_base, ]
            # avoid extremes, or flux will be invalid
            lower_T_lim = 50
            upper_T_lim = 1e10
            if 'Sorg' in self.name:
                lower_T_lim = 10
                upper_T_lim = 500

            # calc temperatures relative to base pressure, from bottom to top of atmosphere
            for i, ln_P_up_i in enumerate(ln_P[1:]): # start after base, T at base already defined
                ln_P_low_i = ln_P[i]
                ln_T_low_i = np.log(temperature[-1])
                # compute temperatures based on gradient
                ln_T_up_i = ln_T_low_i + (ln_P_up_i - ln_P_low_i)*dlnT_dlnP[i+1]
                next_T = np.exp(ln_T_up_i)
                next_T = np.clip(next_T, lower_T_lim, upper_T_lim)
                temperature.append(next_T)

            # reverse order, pRT reads temps from top to bottom of atm
            self.temperature = np.array(temperature[::-1])

        elif self.PT_type=='PTguillot':
            T_int = self.params['T_int']
            T_equ = self.params['T_equ']
            kappa_IR = 10**self.params['log_k_IR']
            gamma = 10**self.params['log_gamma']
            self.gravity = 10**self.params['log_g']
            self.temperature = guillot_global(self.pressure, kappa_IR, gamma, 
                                              self.gravity, T_int, T_equ)

        return self.temperature

    def instrumental_broadening(self, wave, flux, resolution=100000, fwhm=None):

        IB = InstrumentalBroadening(wave, flux)
        if isinstance(resolution, np.ndarray):
            # Variable resolution profile
            flux_LSF = IB(fwhm=const.c.to(u.km/u.s).value/resolution, kernel='gaussian_variable')
            return flux_LSF
        else:
            # Constant resolution
            if fwhm==None: 
                flux_LSF = IB(res=resolution, kernel='gaussian')
            else: # fwhm in km/s
                flux_LSF = IB(fwhm=fwhm, kernel='gaussian')
            return flux_LSF
    
    def make_spectrum_continuous(self,ref_wave): # just for plotting

        file=pathlib.Path(f'atmosphere_objects_continuous.pickle')
        if file.exists():
            atmosphere = load_pickle(file)

        if self.cloud_mode == 'gray': # Gray cloud opacity
            self.give_absorption_opacity=self.gray_cloud_opacity # fsed_gray only needed here, not in calc_flux

        atmosphere.calc_flux(self.temperature,
                        self.mass_fractions,
                        self.gravity,
                        self.MMW,
                        Kzz=self.Kzz, # only for MgSiO3 clouds
                        fsed = self.fsed, # only for MgSiO3 clouds 
                        sigma_lnorm = self.sigma_lnorm, # only for MgSiO3 clouds
                        add_cloud_scat_as_abs=self.add_cloud_scat_as_abs, # only for MgSiO3 clouds
                        contribution =self.contribution,
                        give_absorption_opacity=self.give_absorption_opacity)

        wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
        flux=atmosphere.flux

        # RV+bary shifting and rotational broadening
        wl_shifted= wl*(1.0+(self.params['rv']-self.vbary)/const.c.to('km/s').value)
        waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
        spec = np.interp(waves_even, wl_shifted, flux)
        spec = fastRotBroad(waves_even, spec, self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
        spec = self.instrumental_broadening(waves_even, spec, self.spectral_resolution)
        flux = np.interp(ref_wave, waves_even*1e3, flux) # pRT wavelengths from microns to nm

        return flux
    
    def pRT_to_photon_flux(self,atmosphere):

        nu = atmosphere.freq * u.Hz # Frequency grid from pRT
        wl_um = (const.c / nu).to(u.um)
        wl_m = wl_um.to(u.m)
        # pRT output is per Hz, convert to per micron:
        flux_nu = atmosphere.flux * u.erg / (u.cm**2 * u.s * u.Hz)

        # dν/dλ = -c / λ²  ⇒ |dν/dλ| = c / λ²
        dnu_dlambda = (const.c / wl_m**2).to(u.Hz / u.um)
        flux_lambda = (flux_nu * dnu_dlambda).to(u.erg / (u.cm**2 * u.s * u.um))

        E_photon = (const.h * const.c / wl_m).to(u.J)
        flux_J_m2_s_um = flux_lambda.to(u.J / u.m**2 / u.s / u.um) # Convert flux to J/m²/s/μm

        # Photon flux [photons / s / m² / μm]
        flux_photon = (flux_J_m2_s_um / E_photon).to(1 / (u.s * u.m**2 * u.um))

        R_earth = 6.371e6  # meters
        R_p = 2.61 * R_earth * u.m
        d_p = (5 * u.pc).to(u.m)

        scaling = (R_p / d_p) ** 2
        flux_observed = flux_photon*scaling*4*np.pi
        t_obs = 24 * 3600  # seconds
        flux_observed*= t_obs
        return wl_um.value,flux_observed