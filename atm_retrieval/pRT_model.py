import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from PyAstronomy.pyasl import fastRotBroad
from astropy import constants as const
from astropy import units as u
import pandas as pd
import copy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import pathlib
from scipy.optimize import nnls
import gc
from cloud_cond import simple_cdf_MgSiO3,return_XMgSiO3
from utils import *
import warnings
import re
from scipy.linalg import LinAlgWarning
from scipy.integrate import quad
from scipy.integrate import simps
from astropy.constants import sigma_sb
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
                 retr_obj,
                 contribution=False, # only for plotting atmosphere.contr_em
                 interpolate=True,
                 leave_out=[]):
        
        inherit_attributes = ['primary_label','data_wave','instrument','species_pRT','name',
                              'chemistry','atmosphere_objects','n_atm_layers','species_info',
                              'pressure','PT_type','cloud_mode','spectral_resolution',
                              'mask_isfinite','data_flux','partialP']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(retr_obj, attr))

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
        self.interpolate=interpolate
        #from LIFE_pRT import LIFE_dict
        #for i in range(5):
            #self.params[f'dlnT_dlnP_{i}'] = LIFE_dict[f'dlnT_dlnP_{i}'][0]
        #self.params['T0'] = LIFE_dict['T0'][0]
        self.temperature = self.make_pt() #P-T profile
        self.vbary = retr_obj.target.vbary
        self.gravity = 10**self.params['log_g']

        #if retr_obj.target.name in ['Sorg1X','Sorg20X']:
            #psg_temperature = PSG_input(retr_obj.target.name).temperature#[::2]
            #psg_pressure = PSG_input(retr_obj.target.name).pressure
            #self.temperature = np.interp(self.pressure, psg_pressure, psg_temperature)
            #self.gravity = 10**3.09
            
        self.give_absorption_opacity=None
        self.int_opa_cloud = np.zeros_like(self.pressure)
        if self.instrument=='LIFE' and 'emcont_fraction' in retr_obj.parameters.params:
            self.contribution=True
        else:
            self.contribution=contribution
        self.leave_out = leave_out if leave_out is not list else list(leave_out) # leave out certain species in equchem for CCF
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
            #self.rk_func = lambda x: 10**self.params['log_k_rk']*np.array(x-wl_mid) + 2.

        if self.chemistry=='freechem': # use free chemistry with defined VMRs
            if self.partialP:
                self.mass_fractions, self.CO, self.FeH = self.free_chemistry_partialP(self.species_pRT,self.params)
                self.MMW = self.mass_fractions['MMW']
                self.VMR_dict = self.get_VMR_dict(self.mass_fractions)
            else:
                self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
        elif self.chemistry=='varchem':   
            if self.partialP:
                self.mass_fractions, self.CO, self.FeH = self.var_chemistry_partialP(self.species_pRT,self.params)
            else:
                self.mass_fractions, self.CO, self.FeH = self.var_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
            self.VMR_dict = self.get_VMR_dict(self.mass_fractions)

        elif self.chemistry in ['equchem','quequchem','flexequ']: # use equilibium chemistry
            self.species_hill = retr_obj.species_hill
            self.mass_fractions = self.equ_chemistry(self.species_pRT,self.params)
            # update mass_fractions with isotopolog ratios
            if any(key in self.params for key in ['13CO','C17O','C18O','H2(18)O','log_C12_13_ratio','log_O16_18_ratio','log_H2O16_18_ratio','log_O16_17_ratio']):
                self.mass_fractions = self.get_isotope_mass_fractions(self.species_pRT,self.mass_fractions,self.params) 
            self.MMW = self.mass_fractions['MMW']
            # get new VMR dict, updated with isotopologs
            self.VMR_dict = self.get_VMR_dict(self.mass_fractions)

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
    
    def get_isotope_mass_fractions(self,species,mass_fractions,params):
        #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/chemistry.py
        mass_ratio_13CO_12CO = self.read_species_info('13CO','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C18O_C16O = self.read_species_info('C18O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C17O_C16O = self.read_species_info('C17O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_H218O_H2O = self.read_species_info('H2(18)O','mass')/self.read_species_info('H2O','mass')
        self.C13_12_ratio = 10**(-params.get('log_C12_13_ratio',15))
        self.O18_16_ratio = 10**(-params.get('log_O16_18_ratio',15))
        self.H2O18_16_ratio = 10**(-params.get('log_H2O16_18_ratio',15))
        self.O17_16_ratio = 10**(-params.get('log_O16_17_ratio',15))

        isotopes = ['13CO','C17O','C18O','H2(18)O']
        ratios = ['C13_12_ratio','O17_16_ratio','O18_16_ratio','H2O18_16_ratio']
        for i,isotope in enumerate(isotopes): # for cross-correlation
            if isotope in self.leave_out:
                setattr(self, ratios[i], 0)       

        for species_i in species:
            if (species_i in ['CO_main_iso','CO_high']): # 12CO mass fraction
                CO_linelist = species_i
                mass_fractions[species_i]=(1-self.C13_12_ratio*mass_ratio_13CO_12CO
                                            -self.O18_16_ratio*mass_ratio_C18O_C16O
                                            -self.O17_16_ratio*mass_ratio_C17O_C16O)*mass_fractions[CO_linelist]
                continue
            if (species_i in ['CO_36','CO_36_high']): # 13CO mass fraction
                mass_fractions[species_i]=self.C13_12_ratio*mass_ratio_13CO_12CO*mass_fractions[CO_linelist]
                continue
            if (species_i in ['CO_28','CO_28_high_Sam']): # C18O mass fraction
                mass_fractions[species_i]=self.O18_16_ratio*mass_ratio_C18O_C16O*mass_fractions[CO_linelist]
                continue
            if (species_i in ['CO_27','CO_27_high_Sam']): # C17O mass fraction
                mass_fractions[species_i]=self.O17_16_ratio*mass_ratio_C17O_C16O*mass_fractions[CO_linelist]
                continue
            if (species_i in ['H2O_main_iso','H2O_pokazatel_main_iso']): # H2O mass fraction
                H2O_linelist=species_i
                mass_fractions[species_i]=(1-self.H2O18_16_ratio*mass_ratio_H218O_H2O)*mass_fractions[H2O_linelist]
                continue
            if (species_i=='H2O_181_HotWat78'): # H2_18O mass fraction
                mass_fractions[species_i]=self.H2O18_16_ratio*mass_ratio_H218O_H2O*mass_fractions[H2O_linelist]
                continue
            
        return mass_fractions
    
    # https://github.com/samderegt/retrieval_base/blob/Restructuring/retrieval_base/model_components/chemistry.py
    def equ_chemistry(self,species_pRT,params):

        def load_interp_tables():
            import h5py
            def load_hdf5(file, key):
                with h5py.File(f'{path_tables}/{file}', 'r') as f:
                    return f[key][...]
                
            # Load the interpolation grid (ignore N/O)
            self.P_grid = load_hdf5('grid.hdf5', 'P')
            self.T_grid = load_hdf5('grid.hdf5', 'T')
            self.CO_grid  = load_hdf5('grid.hdf5', 'C/O')
            self.FeH_grid = load_hdf5('grid.hdf5', 'Fe/H')
            points = (self.P_grid, self.T_grid, self.CO_grid, self.FeH_grid)

            from scipy.interpolate import RegularGridInterpolator
            self.interp_tables = {}
            for species_i, hill_i in zip([*species_pRT, 'MMW'], [*self.species_hill, 'MMW']):
                key = 'MMW' if species_i=='MMW' else 'log_VMR'
                equ_table = pathlib.Path(f'{path_tables}/{hill_i}.hdf5')
                if equ_table.exists():
                    arr = load_hdf5(f'{hill_i}.hdf5', key=key)  # Load equchem abundance tables
                else:
                    arr=np.ones_like(load_hdf5('C1O1.hdf5', key=key))*-15
                
                # Generate interpolation functions
                self.interp_tables[species_i] = RegularGridInterpolator(
                    values=arr[:,:,:,0,:], points=points, method='linear', # arr[P,T,C/O,N/O (const, solar value),FeH]
                    #bounds_error=False, fill_value=None
                        )        
                
        def get_VMRs(ParamTable):
            self.VMRs = {}
            self.VMRs = {'He':0.15*np.ones(self.n_atm_layers)}

            def apply_bounds(val, grid):
                val=np.array(val)
                val[val > grid.max()] = grid.max()
                val[val < grid.min()] = grid.min()
                return val

            # Update the parameters
            self.CO  = ParamTable.get('C/O')
            self.FeH = ParamTable.get('Fe/H')

            # Apply the bounds of the grid
            P = apply_bounds(self.pressure.copy(), grid=self.P_grid)
            T = apply_bounds(self.temperature.copy(), grid=self.T_grid)
            CO  = apply_bounds(np.array([self.CO]).copy(), grid=self.CO_grid)[0]
            FeH = apply_bounds(np.array([self.FeH]).copy(), grid=self.FeH_grid)[0]
            
            # Interpolate abundances
            for pRT_name_i, interp_func_i in self.interp_tables.items():

                # Interpolate the equilibrium abundances
                arr_i = interp_func_i(xi=(P, T, CO, FeH))
                if pRT_name_i != 'MMW':
                    
                    species_i = self.species_info[self.species_info["pRT_name"] == pRT_name_i].index[0]

                    if self.chemistry=='flexequ' and species_i not in ['13CO','C17O','C18O','H2(18)O']: # vary equchem by factor
                        vmr = (10**arr_i) *(10**params[f'log_a_{species_i}'])
                        self.VMRs[species_i] =np.clip(vmr, a_min=None, a_max=0.1)
                    else:
                        self.VMRs[species_i] = 10**arr_i # log10(VMR)

                    if species_i in self.leave_out:
                        self.VMRs[species_i].fill(0)
                else:
                    self.MMW = arr_i.copy() # Mean-molecular weight

        def VMR_to_MF():
            MMW = 0.
            for species_i, VMR_i in self.VMRs.items():
                mass_i = self.read_species_info(species_i, 'mass')
                MMW += mass_i * VMR_i

            # Convert to mass-fractions using mass-ratio
            self.mass_fractions = {'MMW': MMW * np.ones(self.n_atm_layers)}
            for species_i, VMR_i in self.VMRs.items():            
                species_pRT_i = self.read_species_info(species_i, 'pRT_name')
                mass_i = self.read_species_info(species_i, 'mass')
                mf = VMR_i * mass_i/MMW
                mf = np.clip(mf, a_min=1e-15, a_max=0.2)
                self.mass_fractions[species_pRT_i] = mf

        def get_H2(): # get H2 abundance as the remainder of the total VMR

            VMR_wo_H2 = np.sum([VMR_i for VMR_i in self.VMRs.values()], axis=0)
            self.VMRs['H2'] = 1 - VMR_wo_H2

            if (self.VMRs['H2'] < 0).any():
                # Other species are too abundant
                print('\nOther species are too abundant')
                if 'H' in self.VMRs.keys(): # was an issue with H
                    print('issue with H',self.VMRs['H'])
                    self.VMRs['H'] *= 1e-3
                    print(self.VMRs['H'])
                else:
                    for species_i in self.VMRs.keys():
                        self.VMRs[species_i] *= 1e-1 # to make phyiscal sense for now

        load_interp_tables()
        get_VMRs(params)
        get_H2()
        VMR_to_MF()

        if self.chemistry=='quequchem':
            for species in self.mass_fractions.keys():
                if any(sub in species for sub in ['H2O_','CO_','CH4_']):
                    Pqu=10**self.params['log_Pqu_CO_CH4'] # is in log
                    idx=find_nearest(self.pressure,Pqu)
                    quenched_fraction=self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx]=quenched_fraction
                elif 'NH3_' in species:
                    Pqu=10**self.params[f'log_Pqu_NH3'] # is in log
                    idx=find_nearest(self.pressure,Pqu)
                    quenched_fraction=self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx]=quenched_fraction
                elif 'HCN_' in species:
                    Pqu=10**self.params[f'log_Pqu_HCN'] # is in log
                    idx=find_nearest(self.pressure,Pqu)
                    quenched_fraction=self.mass_fractions[species][idx]
                    self.mass_fractions[species][:idx]=quenched_fraction
        return self.mass_fractions
    
    def free_chemistry(self,species_pRT,params):
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

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
        #vmr_maxcont = load_pickle('./LIFE/Sorg20X/VMR_maxcont.pickle')
        psg = PSG_input('Sorg20X').table
        
        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            if line_species_i in line_species and line_species_i!='H2':
                vmrs_var = []
                for kn in range(n_knots):
                    vmrs_var.append(VMRs_list[kn][line_species_i])
                #log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=n_knots)
                log_P_knots = generate_skewed_p_nodes(n_nodes=n_knots, skew=0.4)
                # use linear interpolation to avoid going into negative values cubic spline did that)
                log_vmrs=np.interp(np.log10(self.pressure), log_P_knots, np.log10(vmrs_var)) # interpolate for all layers

                ## remove later ####
                #log_vmrs= vmr_maxcont[f'log_{species_i}']*np.ones_like(self.pressure)
                #print(psg[species_i].values[::2])
                #vmr = psg[species_i].values[::2]
                #plt.plot(vmr[::-1],self.pressure[::-1],color=self.read_species_info(species_i,'color'))
                #plt.yscale('log')
                #plt.xscale('log')
                #plt.gca().invert_yaxis()
                #plt.xlim(1e-10,1e-1)
                #plt.savefig('vmrs_life.png')
                #vmr[vmr == 0] = 1e-20
                #log_vmrs = np.log10(vmr)

                ##############################

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

    def free_chemistry_partialP(self, species_pRT, params):
        """
        Free chemistry using retrieved log partial pressures [bar] for ALL species, 
        including H2 and He. The total pressure in each layer is reconstructed from the sum.
        """

        mass_fractions = {}
        #self.VMRs = {} # save for plotting
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
            #print('\nP_tot > P_surf!', np.round(P_tot,decimals=2), ">", max(self.pressure))
            self.unphysical_params = True

        self.P_tot = P_tot

        # Convert partial pressures to VMR and compute mass fractions
        VMR_tot = 0.0
        for species_i, P_i in P_partials.items():
            #P_i = 1e-10 if species_i=='H2O' else P_i
            VMR_i = P_i / P_tot
            VMR_tot += VMR_i
            #print(f"{species_i} {np.log10(P_i):.2f} {np.log10(VMR_i):.2f}")

            #self.VMRs[f'log_{species_i}'] = np.log10(VMR_i)
            species_pRT_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i = self.read_species_info(species_i, 'COH')

            mass_fractions[species_pRT_i] = mass_i * VMR_i
            C += COH_i[0] * VMR_i
            O += COH_i[1] * VMR_i
            H += COH_i[2] * VMR_i

        #print('VMR_tot=',VMR_tot,'\n')
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

    def var_chemistry_partialP(self, line_species, params):
        CO_list = []
        FeH_list = []
        line_species.append('He')
        line_species.append('H2')
        
        n_knots = sum(1 for key in params if re.fullmatch(r'log_p_H2O_\d+', key))  # knots for vertical retrieval
        
        partialP_list = []
        for knot in range(n_knots):
            partialP_sum = 0
            partialP_knot = {}
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

                partialP_knot[species_i] = p_i  # <-- use species_i here for VMR keys
                partialP_sum += p_i

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
            partialP_list.append(partialP_knot)

        # Interpolate partial pressures over layers:
        partialP_interp = {}
        mass_fractions_interp = {}

        # Append He to line_species if not present to interpolate
        if 'He' not in line_species:
            line_species.append('He')

        # Generate knot locations in log pressure space:
        log_P_knots = generate_skewed_p_nodes(n_nodes=n_knots, skew=0.4)

        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            if line_species_i in line_species:
                p_knots = []
                for kn in range(n_knots):
                    # Here partialP_list uses species_i keys
                    p_knots.append(partialP_list[kn].get(species_i, 0))
                
                # interpolate log partial pressures over atmospheric layers
                # add a tiny floor to avoid log(0)
                p_knots_safe = np.array(p_knots)
                p_knots_safe[p_knots_safe == 0] = 1e-30
                log_p_knots = np.log10(p_knots_safe)
                log_pressure_layers = np.log10(self.pressure)

                log_p_interp = np.interp(log_pressure_layers, log_P_knots, log_p_knots)
                partialP_interp[species_i] = 10 ** log_p_interp  # <-- key species_i for VMRs

        # Calculate total pressure per layer by summing partial pressures
        partialP_array = np.vstack([partialP_interp[spec] for spec in partialP_interp])
        P_tot_layers = np.sum(partialP_array, axis=0)

        # Calculate VMRs = p_i / P_tot per layer (keys species_i)
        VMRs_interp = {}
        for spec in partialP_interp:
            VMRs_interp[spec] = partialP_interp[spec] / P_tot_layers

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
    
    def make_spectrum(self):
        
        self.norm_factor_A = None
        self.norm_factor_B = None
        summed_emconts =[]

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

            # MgSiO3 cloud model like in Sam's code
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

            #plt.plot(self.temperature,self.pressure)
            #plt.yscale('log')
            #plt.gca().invert_yaxis()
            #plt.savefig('pt_life2.jpg')
            #print('gravity',np.log10(self.gravity))
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

            #wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
            #wl = (const.c / (atmosphere.freq * u.GHz)).to(u.cm)
            
            wl_cm = const.c.to(u.km/u.s).value/atmosphere.freq/1e-5 # cm
            # [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
            flux = atmosphere.flux*const.c.to(u.km/u.s).value/(wl_cm**2) # convert from flux density to flux
            #print('max flux',np.max(flux))
            wl = wl_cm/1e-4 # microns
            #print([a for a in dir(atmosphere) if not a.startswith('_')])

            #if self.contribution==False:
                #self.summed_emcont = self.estimate_emcontibution_function(atmosphere,wl_cm)
            
            if False: #self.primary_label==False: # calc continuum
                zero_mf = {key: np.zeros_like(val) for key, val in self.mass_fractions.items()}
                atmosphere2= copy.deepcopy(atmosphere)
                atmosphere2.calc_flux(self.temperature, zero_mf, self.gravity, self.MMW, contribution =False)
                continuum_flux = atmosphere.flux*const.c.to(u.km/u.s).value/(wl**2)
                waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
                wl_shifted= wl*(1.0+(self.params['rv']-self.vbary)/const.c.to('km/s').value)
                continuum_flux = np.interp(waves_even, wl_shifted, continuum_flux)
                ref_wave = self.data_wave.reshape(7,3*2048)[part]# [nm]
                continuum_flux = np.interp(ref_wave, waves_even*1e3, continuum_flux)
                self.continuum_flux[part] = continuum_flux

            if self.contribution==True: # emission contribution
                #contr_estimated = self.estimate_contribution_function(atmosphere,wl_cm)
                #plt.plot(contr_estimated/np.max(contr_estimated),self.pressure)
                self.summed_emcont = np.nansum(atmosphere.contr_em,axis=1) # sum over all wavelengths
                summed_emconts.append(self.summed_emcont)
                #plt.plot(self.summed_emcont/np.max(self.summed_emcont),self.pressure)
                #plt.yscale('log')
                #plt.gca().invert_yaxis()
                #plt.savefig('emcont.jpg')

            if self.instrument=='CRIRES':
                # RV+bary shifting and rotational broadening
                waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
                wl_shifted= wl*(1.0+(self.params['rv']-self.vbary)/const.c.to('km/s').value)
                flux = np.interp(waves_even, wl_shifted, flux)
                flux = fastRotBroad(waves_even, flux, self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
                flux = self.instrumental_broadening(waves_even, flux, self.spectral_resolution)

            if self.instrument=='LIFE':

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
                
                if False:
                    print(f"Planck flux (1–30 μm): {B_int_full:.3e}")
                    print(f"Planck flux (4–18 μm): {B_int_partial:.3e}")
                    print(f"Flux fraction: {flux_fraction:.3f}")
                    print(f"Teff = {Teff:.1f} K")
                    print(f"Teff (corrected) = {Teff_corrected:.1f} K")

                    plt.figure()
                    plt.plot(wl_full_um, B_full, label='B_lambda (280K)', lw=1.5)
                    plt.fill_between(wl_model_um, planck_lambda_um(T_ref, wl_model_um), alpha=0.3, label='Your λ-range')
                    plt.xlabel('Wavelength [μm]')
                    plt.ylabel('B_lambda [erg / cm² / s / μm / sr]')
                    plt.title('Planck Spectrum and Your Wavelength Range')
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig('debugging2.jpg')
   
                wl_um,flux_observed = self.pRT_to_photon_flux(atmosphere)
                #print('max before broad',np.nanmax(flux_observed))
                flux_observed = self.instrumental_broadening(wl_um,flux_observed,50)
                
                #flux_observed = self.convolve_to_resolution(wl_um,flux_observed,50)
                #print('max after broad',np.nanmax(flux_observed))
                flux_model = np.interp(self.data_wave.flatten(), wl_um, flux_observed)
                #print('max after interp',np.nanmax(flux_model))
                flux_model = flux_model.reshape(self.data_flux.shape)

                #plt.plot(self.data_wave.flatten(),self.data_flux.flatten(),c='k')
                #plt.plot(self.data_wave.flatten(),flux_model.flatten(),c='r')
                #plt.savefig('life_model.jpg',dpi=200)
                #print('maxima',np.max(flux_model),np.max(self.data_flux),np.max(flux_model)/np.max(self.data_flux))

                if np.nanmax(flux_model) in [0, np.nan, np.inf] or len(flux_model)==0 or self.unphysical_params==True:
                    #raise ZeroDivisionError('Invalid flux',np.nanmax(flux_model))
                    print('\nInvalid flux, max=',np.nanmax(flux_model))
                    return np.ones_like(self.data_flux)*np.nanmedian(self.data_flux)
                else:
                    return flux_model #flux/np.nanmax(flux)

            # Interpolate/rebin onto the data's wavelength grid
            # should not be done when making spectrum for cross-corr, or wl padding will be cut off
            if self.interpolate==True:
                ref_wave = data_wave_orders[part]# [nm]
                flux = np.interp(ref_wave, waves_even*1e3, flux) # pRT wavelengths from microns to nm

            if self.primary_label==False and self.interpolate==True: # should have same wavelengths
                order=part
                
                flux = flux.reshape(self.n_dets,self.n_pixels)
                for det in range(3):

                    nonans = np.isfinite(self.primary_flux[order][det]) & np.isfinite(self.data_flux[order][det]) & np.isfinite(self.data_err[order][det])
                    wl_det = self.data_wave.reshape(self.n_orders,self.n_dets,self.n_pixels)[order,det]
                    
                    if 'log_k_rk' in self.params: # veiling
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
                    #plt.plot(wl_det,flux[det],c='b')
                    #plt.plot(wl_det[nonans],star,c='r')
                    #plt.savefig('modelAB_norm.jpg')
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
            return spectrum_parts, waves_parts
        else:
            spectrum_parts=np.array(spectrum_parts)
            #np.savetxt('physical_spectrum_A.txt', spectrum_parts.flatten())
            spectrum_parts = spectrum_parts.reshape(data_shape)
            if self.primary_label==False:
                data_flux = self.data_flux.reshape(data_shape)
                data_wave = self.data_wave.reshape(data_shape)
                self.continuum_flux = self.continuum_flux.reshape(data_shape)
                for i,part in enumerate(spectrum_parts):
                    if np.isnan(data_flux[i]).all():
                        continue
                    #spectrum_parts[i] = upper_envelope_remove_continuum(data_wave[i],spectrum_parts[i],oneD=True)
                    spectrum_parts[i] = fft_remove_continuum(spectrum_parts[i])
                    #spectrum_parts[i] /=np.nanmedian(spectrum_parts[i])
                    #spectrum_parts[i] *=np.nanmedian(data_flux[i])
                return spectrum_parts
            else: # normalize in same way as data spectrum
                if self.name in ['ROXs12A','ROXs12_onlyB']: # normalized differently
                    for i,part in enumerate(spectrum_parts):
                        if 'log_k_rk' in self.params:
                            spectrum_parts[i] /=np.nanpercentile(spectrum_parts[i],97)
                            rk = self.rk_func(self.data_wave[i])
                            sp = spectrum_parts[i]
                            spectrum_parts[i] = (sp+rk*np.nanmedian(sp))/(1+rk)
                        #spectrum_parts[i] = upper_envelope_remove_continuum(self.data_wave[i],spectrum_parts[i])
                        spectrum_parts[i] = fft_remove_continuum(spectrum_parts[i])
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
                #print("self.params['log_P_mid']",self.params['log_P_mid'])
                #print("self.params['log_delta_P']",self.params['log_delta_P'])
                #print('n_iter',n_iter)
                for n in range(n_iter):
                    n+=1
                    p_below = np.log10(10**self.params['log_P_RCB']/10**self.params['log_delta_P'])
                    p_above = np.log10(10**self.params['log_P_RCB']*10**self.params['log_delta_P'])
                    #print(central_n-n,central_n+n)
                    #print('p_below',p_below,np.log10(10**(log_max_p)*0.9))
                    #print('p_above',p_above,np.log10(10**(log_min_p)*1.1))
                    self.log_P_knots[central_n-n] = max(np.log10(10**(log_min_p)*1.1), p_below)
                    self.log_P_knots[central_n+n] = min(np.log10(10**(log_max_p)*0.9), p_above)
                 
                if n_grad%2==0: # even number of PT points
                    p_below = np.log10(10**self.params['log_P_RCB']-2*10**(self.params['log_delta_P']))
                    #print('new p_below',p_below)
                    n+=1 # n from previous loop
                    self.log_P_knots[central_n-n] = max(log_min_p, p_below)

                
                #print("self.log_P_knots",self.log_P_knots)

                #for n in range(n_grad):
                    #self.log_P_knots.append(self.params[f'log_P_{n}'])
                #self.log_P_knots = np.array(self.log_P_knots)

            else: # use uniformly spaced PT-knots
                self.log_P_knots = np.linspace(np.log10(np.max(self.pressure)),
                                            np.log10(np.min(self.pressure)),
                                            num=n_grad)

            if 'Sorg' in self.name: # denser at bottom
                self.log_P_knots = generate_skewed_p_nodes(n_nodes=n_grad, skew=0.4)[::-1]
                #self.log_P_knots = [0,-0.5,-1.5,-2.5,-6]

            if 'dlnT_dlnP_knots' not in kwargs:
                self.dlnT_dlnP_knots=[]
                for i in range(n_grad):
                    self.dlnT_dlnP_knots.append(self.params[f'dlnT_dlnP_{i}'])
            elif 'dlnT_dlnP_knots' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                self.dlnT_dlnP_knots=kwargs.get('dlnT_dlnP_knots')

            #print('P and Tgrad at RCB:',self.log_P_knots[central_n],self.dlnT_dlnP_knots[central_n])
            #print('self.log_P_knots\n',self.log_P_knots)
            #print('self.dlnT_dlnP_knots\n',self.dlnT_dlnP_knots)

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
                lower_T_lim = 50
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
            self.temperature = np.array(temperature[::-1]) # reverse order, pRT reads temps from top to bottom of atm
            #plt.plot(self.temperature,self.pressure)
            #plt.yscale('log')
            #plt.ylim(max(self.pressure),min(self.pressure))
            #plt.savefig('life_pt.png')
        elif self.PT_type=='PTguillot':
            T_int = self.params['T_int']
            T_equ = self.params['T_equ']
            kappa_IR = 10**self.params['log_k_IR']
            gamma = 10**self.params['log_gamma']
            self.gravity = 10**self.params['log_g']
            self.temperature = guillot_global(self.pressure, kappa_IR, gamma, self.gravity, T_int, T_equ)

        #plt.plot(self.temperature,self.pressure)
        #interp_func = interp1d(np.log10(self.pressure), self.temperature, kind='linear', fill_value="extrapolate")
        #T_knots = interp_func(self.log_P_knots)
        #plt.scatter(T_knots,10**self.log_P_knots,s=10,c='r')
        #plt.yscale('log')
        #plt.gca().invert_yaxis()
        #plt.savefig('pt.jpg')
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
    
    def make_spectrum_continuous(self,ref_wave): # just for plotting, not needed for retrieval

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
        #flux/=np.nanmedian(flux)

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

    def estimate_contribution_function(self, atmosphere, wl_cm):
        """
        Computes the wavelength-integrated emission contribution function
        for a petitRADTRANS atmosphere object (pRT v2.7).
        """
        P_cgs = atmosphere.press
        T = atmosphere.temp
        MMW = atmosphere.mmw

        if P_cgs[0] > P_cgs[-1]:
            P_cgs = P_cgs[::-1]
            T = T[::-1]
            MMW = MMW[::-1]

        _, species_kappas = atmosphere.get_opa(np.array([T]))
        total_kappa = np.zeros_like(next(iter(species_kappas.values())))  # (nwavelengths, nlayers)
        for sp in species_kappas:
            total_kappa += species_kappas[sp]
        kappa = total_kappa.T  # (nlayers, nwavelengths)

        # --- Density ρ ---
        R_cgs = 8.314462618e7   # erg/mol/K
        rho = (P_cgs * MMW) / (R_cgs * T)   # g/cm^3

        # --- dz from hydrostatic equilibrium ---        
        dP = np.diff(P_cgs)
        P_mid = 0.5 * (P_cgs[:-1] + P_cgs[1:])
        rho_mid = 0.5 * (rho[:-1] + rho[1:])
        dz_mid = dP / (rho_mid * self.gravity)

        dz = np.empty_like(P_cgs)
        dz[1:-1] = 0.5 * (dz_mid[1:] + dz_mid[:-1])
        dz[0] = dz_mid[0]
        dz[-1] = dz_mid[-1]

        plt.figure()
        plt.plot(dz, P_cgs)
        plt.gca().invert_yaxis()
        plt.yscale('log')
        plt.xlabel('dz [cm]')
        plt.ylabel('Pressure [cgs]')
        plt.title('Vertical Layer Thickness')
        plt.savefig('dz.jpg')
        plt.close()

        # --- Optical depth ---
        delta_tau = kappa * (rho[:, None] * dz[:, None])  # (nlayers, nwavelengths)
        tau = np.cumsum(delta_tau, axis=0)
        tau_shifted = np.vstack([np.zeros((1, tau.shape[1])), tau[:-1, :]])

        # --- Planck weighting ---
        wl_m = wl_cm[:, None] * 1e-2  # (nwavelengths, 1)
        B_lambda = (2 * sc.h * sc.c**2 / wl_m**5) / (
            np.exp(sc.h * sc.c / (wl_m * sc.k * T[None, :])) - 1
        )  # shape: (nwavelengths, nlayers)
        B_lambda = B_lambda.T  # shape: (nlayers, nwavelengths)

        # --- Contribution function ---
        #d_exp_tau = np.exp(-tau_shifted) * (1 - np.exp(-delta_tau)) * B_lambda  # shape: (nlayers, nwavelengths)
        mu = 0.5
        tau_mu = tau / mu
        K = tau_mu * np.exp(-tau_mu)
        K /= np.trapz(K, axis=0)
        from scipy.ndimage import gaussian_filter1d
        B_lambda_smoothed = gaussian_filter1d(B_lambda, sigma=5, axis=1)

        d_exp_tau = K * B_lambda_smoothed

        # --- Integrate over wavelengths ---
        emission = np.trapz(d_exp_tau, x=wl_cm, axis=1)  # shape: (nlayers,)

        # --- Weight by log(P) bin size ---
        logP = np.log(P_cgs)
        dlogP = np.diff(logP, append=logP[-1])
        dlogP[-1] = dlogP[-2]  # avoid repeated last bin
        contribution = emission * dlogP

        # --- Normalize contribution ---
        contribution /= np.sum(contribution)

        # --- Find peak pressure ---
        max_index = np.argmax(contribution)
        P_bar = P_cgs * 1e-6
        P_peak_bar = P_bar[max_index]

        # --- Plot diagnostic ---
        if False:
            plt.figure(figsize=(6, 4))
            plt.plot(contribution, P_bar)
            plt.gca().invert_yaxis()
            plt.yscale('log')
            plt.xlabel("Contribution")
            plt.ylabel("Pressure [bar]")
            plt.title("Emission Contribution Function")
            plt.tight_layout()
            plt.savefig("contribution_fn.png")
            print(f"Max contribution at log10(P/bar) = {np.log10(P_peak_bar):.2f}")
            print(f"Contribution peak value = {np.max(contribution):.3e}")

        return contribution




