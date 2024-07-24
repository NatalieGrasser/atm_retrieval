# generate synthetic spectrum based on 2M0355 to test retrieval

test_dict={'rv': (11.9,r'$v_{\rm rad}$'),
            'vsini': (5.5,r'$v$ sin$i$'),
            'log_g':(4.75,r'log $g$'),
            'epsilon_limb': (0.6, r'$\epsilon_\mathrm{limb}$'),
            'log_H2O':(-3.4,r'log H$_2$O'),
            'log_12CO':(-3.7,r'log $^{12}$CO'),
            'log_13CO':(-5.2,r'log $^{13}$CO'),
            'log_CH4':(-6.7,r'log CH$_4$'),
            'log_NH3':(-7.1,r'log NH$_3$'),
            'log_HF':(-6.5,r'log HF'),
            'log_H2(18)O':(-7.5,r'log H$_2^{18}$O'),
            'log_H2S':(-5.4,r'log H$_2$S')}

test_parameters={}
test_mathtext={}
for key_i, (value_i, mathtext_i) in test_dict.items():
   test_parameters[key_i]   = value_i
   test_mathtext[key_i] = mathtext_i

# only execute code if run directly from terminal, otherwise just import params dict
if __name__ == "__main__":
      
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import os
   from petitRADTRANS import Radtrans
   from PyAstronomy.pyasl import fastRotBroad, helcorr
   from astropy import constants as const
   from astropy import units as u
   from astropy.coordinates import SkyCoord
   from spectrum import Spectrum, convolve_to_resolution
   from scipy.ndimage import gaussian_filter


   def load_spectrum(target):
      file=np.genfromtxt(f"{target}/{target}_spectrum.txt",skip_header=1,delimiter=' ')
      wl=np.reshape(file[:,0],(7,3,2048))
      fl=np.reshape(file[:,1],(7,3,2048))
      err=np.reshape(file[:,2],(7,3,2048))
      return wl,fl,err

   def get_species(param_dict): # get pRT species name from parameters dict
      species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
      chem_species=[]
      for par in param_dict:
            if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
               if par in ['log_g','log_Kzz','log_P_base_gray','log_opa_base_gray','log_a','log_l']: # skip
                  pass
               else:
                  chem_species.append(par)
      species=[]
      for chemspec in chem_species:
            species.append(species_info.loc[chemspec[4:],'pRT_name'])
      return species

   def read_species_info(species,info_key):
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

   def free_chemistry(line_species,params,n_atm_layers):
      species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
      VMR_He = 0.15
      VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
      mass_fractions = {} # Create a dictionary for all used species
      C, O, H = 0, 0, 0

      for species_i in species_info.index:
         line_species_i = read_species_info(species_i,'pRT_name')
         mass_i = read_species_info(species_i, 'mass')
         COH_i  = read_species_info(species_i, 'COH')

         if species_i in ['H2', 'He']:
               continue
         if line_species_i in line_species:
               VMR_i = 10**(params[f'log_{species_i}'])*np.ones(n_atm_layers) #  use constant, vertical profile

               # Convert VMR to mass fraction using molecular mass number
               mass_fractions[line_species_i] = mass_i * VMR_i
               VMR_wo_H2 += VMR_i

               # Record C, O, and H bearing species for C/O and metallicity
               C += COH_i[0] * VMR_i
               O += COH_i[1] * VMR_i
               H += COH_i[2] * VMR_i

      # Add the H2 and He abundances
      mass_fractions['He'] =read_species_info('He', 'mass')*VMR_He
      mass_fractions['H2'] =read_species_info('H2', 'mass')*(1-VMR_wo_H2)
      H += read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
      
      if VMR_wo_H2.any() > 1:
         print('VMR_wo_H2 > 1. Other species are too abundant!')

      MMW = 0 # Compute the mean molecular weight from all species
      for mass_i in mass_fractions.values():
         MMW += mass_i
      MMW *= np.ones(n_atm_layers)
      
      for line_species_i in mass_fractions.keys():
         mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
      mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
      CO = C/O
      log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
      FeH = np.log10(C/H)-log_CH_solar
      CO = np.nanmean(CO)
      FeH = np.nanmean(FeH)

      return mass_fractions, CO, FeH
   
   def instr_broadening(wave, flux, out_res=1e6, in_res=1e6):

        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2-1/in_res**2)/(2*np.sqrt(2*np.log(2)))
        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter,mode='nearest')

        return flux_LSF


   K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                  [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                  [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                  [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                  [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                  [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                  [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])

   data_wave,data_flux,data_err=load_spectrum('2M0355') # test spectrum based on 2M0355
   n_orders=7
   lbl_opacity_sampling=3

   # sonora bobcat P-T profile (T=1400K, logg=4.65, solar metallicity, solar C/O-ratio)
   file=np.loadtxt('t1400g562nc_m0.0.dat')
   pres=file[:,1] # bar
   temp=file[:,2] # K
   n_atm_layers=len(pres)

   species=get_species(test_parameters)
   mass_fractions, CO, FeH=free_chemistry(species,test_parameters,n_atm_layers)
   gravity = 10**test_parameters['log_g'] 
   coords = SkyCoord(ra="03h55m23.3735910810s", dec="+11d33m43.797034332s", frame='icrs') # use 2M0355

   spectrum_orders=[]
   for order in range(n_orders):
      wl_pad=7 # wavelength padding because spectrum is not wavelength shifted yet
      wlmin=np.min(K2166[order])-wl_pad
      wlmax=np.max(K2166[order])+wl_pad
      wlen_range=np.array([wlmin,wlmax])*1e-3 # nm to microns

      atmosphere = Radtrans(line_species=species,
                        rayleigh_species = ['H2', 'He'],
                        continuum_opacities = ['H2-H2', 'H2-He'],
                        wlen_bords_micron=wlen_range, 
                        mode='lbl',
                        lbl_opacity_sampling=lbl_opacity_sampling)
      
      atmosphere.setup_opa_structure(pres)
      atmosphere.calc_flux(temp,
                           mass_fractions,
                           gravity,
                           mass_fractions['MMW'],
                           contribution=False)
      
      wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
      flux=atmosphere.flux
      #flux = atmosphere.flux/np.nanmean(atmosphere.flux)

      # RV+bary shifting and rotational broadening
      v_bary, _ = helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, # of Cerro Paranal
                        ra2000=coords.ra.value,dec2000=coords.dec.value,jd=2459885.5)
      wl_shifted= wl*(1.0+(test_parameters['rv']-v_bary)/const.c.to('km/s').value)
      spec = Spectrum(flux, wl_shifted)
      waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
      spec = fastRotBroad(waves_even, spec.at(waves_even), test_parameters['epsilon_limb'], test_parameters['vsini'])
      spec = Spectrum(spec, waves_even)
      spec = convolve_to_resolution(spec,100_000)

      #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/spectrum.py#L289
      resolution = int(1e6/lbl_opacity_sampling)
      flux_broad=instr_broadening(spec.wavelengths*1e3,spec,out_res=resolution,in_res=500000)

      # Interpolate/rebin onto the data's wavelength grid
      ref_wave = data_wave[order].flatten() # [nm]
      flux = np.interp(ref_wave, spec.wavelengths*1e3, flux_broad) # pRT wavelengths from microns to nm

      # reshape to (detectors,pixels) so that we can store as shape (orders,detectors,pixels)
      flux=flux.reshape(data_wave.shape[1],data_wave.shape[2])
      spectrum_orders.append(flux)

   test_spectrum=np.array(spectrum_orders)
   test_spectrum/=np.nanmedian(test_spectrum) # normalize in same way as data spectrum
   test_spectrum[np.isnan(data_flux)]=np.nan # mask same regions as in observed data

   # add Gaussian noise by using flux_err*s^2 (approximate mean of s^2)
   test_spectrum_noisy=test_spectrum+np.random.normal(0,np.nanmean(data_err)*10,size=test_spectrum.shape)

   spectrum=np.full(shape=(2048*7*3,3),fill_value=np.nan)
   spectrum[:,0]=data_wave.flatten()
   spectrum[:,1]=test_spectrum_noisy.flatten()
   spectrum[:,2]=data_err.flatten()
   np.savetxt('test/test_spectrum.txt',spectrum,delimiter=' ',header='wavelength (nm) flux flux_error')

   wl,fl,err=load_spectrum('2M0355')
   wlm,flm,errm=load_spectrum('test')
   fig,ax=plt.subplots(1,1,figsize=(9,2),dpi=200)
   ax.plot(wl.flatten(),fl.flatten(),label='2M0355')
   ax.plot(wlm.flatten(),flm.flatten(),label='testspec',alpha=0.5)
   ax.legend()
   ax.set_xlabel('Wavelength [nm]')
   fig.tight_layout(h_pad=0)
   fig.savefig(f'test/test_spectrum.pdf')
   plt.close()