import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
from PyAstronomy.pyasl import fastRotBroad, helcorr
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd

import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import atm_retrieval.cloud_cond as cloud_cond
    from atm_retrieval.cloud_cond import simple_cdf_MgSiO3,return_XMgSiO3
    from atm_retrieval.spectrum import Spectrum, convolve_to_resolution
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
elif getpass.getuser() == "natalie": # when testing from my laptop
    import cloud_cond as cloud_cond
    from cloud_cond import simple_cdf_MgSiO3,return_XMgSiO3
    from spectrum import Spectrum, convolve_to_resolution

class pRT_spectrum:

    def __init__(self,
                 parameters,
                 data_wave, # shape (orders,detectors,pixels)
                 target,
                 species,
                 atmosphere_objects,
                 spectral_resolution=100_000,  
                 cloud_mode=None,contribution=False,free_chem=True): # contribution only for plotting atmosphere.contr_em
        
        self.params=parameters
        self.data_wave=data_wave
        self.target=target
        self.coords = SkyCoord(ra=target.ra, dec=target.dec, frame='icrs')
        self.species=species
        self.spectral_resolution=spectral_resolution
        self.free_chem=free_chem
        self.atmosphere_objects=atmosphere_objects

        self.n_atm_layers=50
        self.pressure = np.logspace(-6,2,self.n_atm_layers)  # like in deRegt+2024
        self.temperature = self.make_pt() #P-T profile

        self.give_absorption_opacity=None
        self.int_opa_cloud = np.zeros_like(self.pressure)
        self.gravity = 10**self.params['log_g'] 
        self.contribution=contribution
        self.cloud_mode=cloud_mode
    
        # do_scat_emis, sigma_lnorm, fsed, Kzz only relevant when there are clouds
        self.sigma_lnorm=None
        self.Kzz=None 
        self.fsed=None

        if self.free_chem==True: # use free chemistry with defined VMRs
            self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species,self.params)
            self.MMW = self.mass_fractions['MMW']

        if self.free_chem==False: # use equilibium chemistry
            abunds = self.abundances(self.pressure,self.temperature,self.params['FEH'],self.params['C_O'])
            self.mass_fractions = self.get_abundance_dict(self.species,abunds)
            self.mass_fractions = self.get_isotope_mass_fractions(self.species,self.mass_fractions,self.params) # update mass_fractions with isotopologue ratios
            self.MMW = abunds['MMW']

        self.spectrum_orders=[]
        self.orders=7

    def abundances(self,press, temp, feh, C_O):
        COs = np.ones_like(press)*C_O
        fehs = np.ones_like(press)*feh
        mass_fractions = interpol_abundances(COs,fehs,temp,press)
        return mass_fractions

    def get_abundance_dict(self,species,abunds): # does not inlcude isotopes
        mass_fractions = {}
        for specie in species:
            if specie in ['H2O_main_iso','H2O_pokazatel_main_iso']:
                mass_fractions[specie] = abunds['H2O']
            #elif specie=='CO_36':
                #mass_fractions[specie] = abunds['13CO']
            elif specie=='CO_main_iso':
                mass_fractions[specie] = abunds['CO']
            elif specie in ['CH4_main_iso','CH4_hargreaves_main_iso']:
                mass_fractions[specie] = abunds['CH4']
            elif specie=='FeH_main_iso':
                mass_fractions[specie] = abunds['FeH']
            elif specie=='HCN_main_iso':
                mass_fractions[specie] = abunds['HCN']
            elif specie=='K':
                mass_fractions[specie] = abunds['K']
            elif specie=='Na_allard':
                mass_fractions[specie] = abunds['Na']
            elif specie=='NH3_coles_main_iso':
                mass_fractions[specie] = abunds['NH3']
            elif specie=='HF_main_iso':
                mass_fractions[specie] = abunds['HF']
            elif specie=='H2S_ExoMol_main_iso':
                mass_fractions[specie] = abunds['H2S']
            elif specie=='OH_main_iso':
                mass_fractions[specie] = abunds['OH']
            elif specie=='CO2_main_iso':
                mass_fractions[specie] = abunds['CO2']
            
        mass_fractions['H2'] = abunds['H2']
        mass_fractions['He'] = abunds['He']
        return mass_fractions
    
    def read_species_info(self,species,info_key):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if info_key == 'pRT_name':
            return species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return species_info.loc[species,'Hill_notation']
        if info_key == 'mass':
            return species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return species_info.loc[species,info_key]
        if info_key == 'c' or info_key == 'color':
            return species_info.loc[species,'color']
        if info_key == 'label':
            return species_info.loc[species,'mathtext_name']
    
    def get_isotope_mass_fractions(self,species,mass_fractions,params):
        #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/chemistry.py
        mass_ratio_13CO_12CO = self.read_species_info('13CO','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C18O_C16O = self.read_species_info('C18O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C17O_C16O = self.read_species_info('C17O','mass')/self.read_species_info('12CO','mass')
        for species_i in species:
            if (species_i=='CO_main_iso'): # 12CO mass fraction
                mass_fractions[species_i]=(1-params['C13_12_ratio']*mass_ratio_13CO_12CO
                                            -params['O18_16_ratio']*mass_ratio_C18O_C16O
                                            -params['O17_16_ratio']*mass_ratio_C17O_C16O)*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_36'): # 13CO mass fraction
                mass_fractions[species_i]=params['C13_12_ratio']*mass_ratio_13CO_12CO*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_28'): # C18O mass fraction
                mass_fractions[species_i]=params['O18_16_ratio']*mass_ratio_C18O_C16O*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_27'): # C17O mass fraction
                mass_fractions[species_i]=params['O17_16_ratio']*mass_ratio_C17O_C16O*mass_fractions['CO_main_iso']
                continue
        return mass_fractions
    
    def free_chemistry(self,line_species,params):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

        for species_i in species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if line_species_i in line_species:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[line_species_i] = mass_i * VMR_i
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
            print('VMR_wo_H2 > 1. Other species are too abundant!')

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for line_species_i in mass_fractions.keys():
            mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar
        CO = np.mean(CO)
        FeH = np.mean(FeH)

        return mass_fractions, CO, FeH

    def gray_cloud_opacity(self): # like in deRegt+2024
        def give_opacity(wave_micron=self.wave_micron,pressure=self.pressure):
            opa_gray_cloud = np.zeros((len(self.wave_micron),len(self.pressure))) # gray cloud = independent of wavelength
            opa_gray_cloud[:,self.pressure>10**(self.params['log_P_base_gray'])] = 0 # [bar] constant below cloud base
            # Opacity decreases with power-law above the base
            above_clouds = (self.pressure<=10**(self.params['log_P_base_gray']))
            opa_gray_cloud[:,above_clouds]=(10**(self.params['log_opa_base_gray']))*(self.pressure[above_clouds]/10**(self.params['log_P_base_gray']))**self.params['fsed_gray']
            if self.params.get('cloud_slope') is not None:
                opa_gray_cloud *= (self.wave_micron[:,None]/1)**self.params['cloud_slope']
            return opa_gray_cloud
        return give_opacity
    
    def make_spectrum(self):

        spectrum_orders=[]
        self.contr_em_orders=[]
        for order in range(self.orders):
            atmosphere=self.atmosphere_objects[order]

            if self.cloud_mode == 'MgSiO3':
                # Cloud model like in Sam's code, mask pressure above cloud deck
                if self.free_chem==True:
                    co=self.CO
                    feh=self.FeH
                if self.free_chem==False:
                    feh=self.params['FEH']
                    co=self.params['C_O']
                P_base_MgSiO3 = simple_cdf_MgSiO3(self.pressure,self.temperature,feh,co,np.mean(self.MMW))
                above_clouds = (self.pressure<=P_base_MgSiO3)
                eq_MgSiO3 = return_XMgSiO3(feh, co)
                self.mass_fractions['MgSiO3(c)'] = np.zeros_like(self.temperature)
                condition=eq_MgSiO3*(self.pressure[above_clouds]/P_base_MgSiO3)**self.params['fsed']
                self.mass_fractions['MgSiO3(c)'][above_clouds]=condition
                self.sigma_lnorm=self.params['sigma_lnorm']
                self.Kzz = 10**self.params['log_Kzz']*np.ones_like(self.pressure)
                self.fsed=self.params['fsed']

            elif self.cloud_mode == 'gray': # Gray cloud opacity
                self.wave_micron = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
                self.give_absorption_opacity=self.gray_cloud_opacity()
                self.Kzz = 10**self.params['log_Kzz']*np.ones_like(self.pressure)
                self.fsed=self.params['fsed_gray']

            atmosphere.calc_flux(self.temperature,
                            self.mass_fractions,
                            self.gravity,
                            self.MMW,
                            Kzz=self.Kzz,
                            fsed = self.fsed, 
                            sigma_lnorm = self.sigma_lnorm, 
                            add_cloud_scat_as_abs=True,
                            contribution =self.contribution,
                            give_absorption_opacity=self.give_absorption_opacity)

            wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
            flux = atmosphere.flux/np.nanmean(atmosphere.flux)

            # Do RV+bary shifting and rotational broadening
            v_bary, _ = helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, # of Cerro Paranal
                            ra2000=self.coords.ra.value,dec2000=self.coords.dec.value,jd=self.target.JD) # https://ssd.jpl.nasa.gov/tools/jdc/#/cd
            wl_shifted= wl*(1.0+(self.params['rv']-v_bary)/const.c.to('km/s').value)
            spec = Spectrum(flux, wl_shifted)
            waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
            spec = fastRotBroad(waves_even, spec.at(waves_even), self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
            spec = Spectrum(spec, waves_even)
            spec = convolve_to_resolution(spec,self.spectral_resolution)
            
            # Interpolate/rebin onto the data's wavelength grid
            ref_wave = self.data_wave[order].flatten() # [nm]
            flux = np.interp(ref_wave, spec.wavelengths*1e3, spec) # pRT wavelengths from microns to nm

            # reshape to (detectors,pixels) so that we can store as shape (orders,detectors,pixels)
            flux=flux.reshape(self.data_wave.shape[1],self.data_wave.shape[2])
            #ref_wave=ref_wave.reshape(data_wave.shape[1],data_wave.shape[2])
            #spec = Spectrum(flux, ref_wave) 
            spectrum_orders.append(flux)

            if self.contribution==True:
                contr_em = atmosphere.contr_em # emission contribution
                summed_contr = np.nansum(contr_em,axis=1) # sum over all wavelengths
                self.contr_em_orders.append(summed_contr)

        return np.array(spectrum_orders)

    def make_pt(self):
        # if pt profile and condensation curve don't intersect, clouds have no effect
        self.t_samp = np.array([self.params['T4'],self.params['T3'],self.params['T2'],self.params['T1']])
        self.p_samp= np.linspace(np.log10(np.nanmin(self.pressure)),np.log10(np.nanmax(self.pressure)),len(self.t_samp))
        sort = np.argsort(self.p_samp)
        temperature = CubicSpline(self.p_samp[sort],self.t_samp[sort])(np.log10(self.pressure))
        return temperature
    

