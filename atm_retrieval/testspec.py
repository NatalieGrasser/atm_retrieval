# generate synthetic spectrum based on 2M0355 to test retrieval

test_dict={'rv': (12.0,r'$v_{\rm rad}$'),
            'vsini': (10.0,r'$v$ sin$i$'),
            'log_g':(4.75,r'log $g$'),
            'epsilon_limb': (0.6, r'$\epsilon_\mathrm{limb}$'),
            'log_H2O':(-3.0,r'log H$_2$O'),
            'log_12CO':(-3.0,r'log $^{12}$CO'),
            'log_13CO':(-5.0,r'log $^{13}$CO'),
            'log_CH4':(-6.0,r'log CH$_4$'),
            'log_HF':(-6.0,r'log HF'),
            'log_H2(18)O':(-6.0,r'log H$_2^{18}$O'),
            'log_H2S':(-5.0,r'log H$_2$S')}
            #'log_a':(2.5,r'$\log\ a$'),
            #'log_a':(0.8,r'$\log\ a$'),
            #'log_l':(-0.5,r'$\log\ l$')}

test_parameters={}
test_mathtext={}
for key_i, (value_i, mathtext_i) in test_dict.items():
   test_parameters[key_i] = value_i
   test_mathtext[key_i] = mathtext_i

# only execute code if run directly from terminal, otherwise just import params dict
if __name__ == "__main__":
      
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import os
   import getpass
   import sys
   import pathlib

   if getpass.getuser() == "natalie": # when testing from my laptop
      os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
      from spectrum import Spectrum, convolve_to_resolution
      from target import Target
      from covariance import *
   elif getpass.getuser() == "grasser": # when running from LEM
      os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"
      from atm_retrieval.spectrum import Spectrum, convolve_to_resolution
      from atm_retrieval.target import Target
      from atm_retrieval.covariance import *
   
   from petitRADTRANS import Radtrans
   from PyAstronomy.pyasl import fastRotBroad, helcorr
   from astropy import constants as const
   from astropy import units as u
   from astropy.coordinates import SkyCoord
   from spectrum import Spectrum, convolve_to_resolution
   from scipy.ndimage import gaussian_filter

   # option to add correlated noise as command line argument
   GP=True if len(sys.argv)>1 else False
   print('Using correlated noise:',GP)

   def get_species(param_dict,chemistry): # get pRT species name from parameters dict
      species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
      if chemistry=='freechem':
         chem_species=[]
         for par in param_dict:
            if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
               if par in ['log_g','log_Kzz','log_P_base_gray','log_opa_base_gray','log_a','log_l',
                            'log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio',
                            'log_Pqu_CO_CH4','log_Pqu_NH3','log_Pqu_HCN']: # skip
                  pass
               else:
                  chem_species.append(par)
         species=[]
         for chemspec in chem_species:
            species.append(species_info.loc[chemspec[4:],'pRT_name'])
      elif chemistry in ['equchem','quequchem']:
         chem_species=['H2O','12CO','13CO','C18O','C17O','CH4','NH3',
                        'HCN','H2(18)O','H2S','CO2','HF','OH'] # HF, OH not in pRT chem equ table
         species=[]
         for chemspec in chem_species:
            species.append(species_info.loc[chemspec,'pRT_name'])
      return species

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
      sigma_LSF = np.sqrt(1/out_res**2-1/in_res**2)/(2*np.sqrt(2*np.log(2)))
      spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))
      sigma_LSF_gauss_filter = sigma_LSF / spacing
      flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter,mode='nearest')
      return flux_LSF

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
      
   def add_RBF_kernel(cov, a, l, separation, variance, trunc_dist=5):
      w_ij = (separation < trunc_dist*l) # Hann window function to ensure sparsity
      GP_amp = a**2 # GP amplitude
      GP_amp *= variance**2
      cov[w_ij] += GP_amp * np.exp(-(separation[w_ij])**2/(2*l**2)) # Gaussian radial-basis function kernel
      return cov
      
   target=Target('2M0355') # test spectrum based on 2M0355
   data_wave,data_flux,data_err=target.load_spectrum()
   mask_isfinite=target.get_mask_isfinite() # mask nans, shape (orders,detectors)
   separation,err_eff=target.prepare_for_covariance()
   K2166=target.K2166
   n_orders,n_dets=target.n_orders,target.n_dets
   lbl_opacity_sampling=3

   # sonora bobcat P-T profile (T=1600K, logg=4.75, solar metallicity, solar C/O-ratio)
   file=np.loadtxt('t1600g562nc_m0.0.dat')
   pres=file[:,1] # bar
   temp=file[:,2] # K
   n_atm_layers=len(pres)

   species=get_species(param_dict=test_parameters,chemistry='freechem')
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

   # random noise depending on data points
   random_noise=np.random.normal(0,np.nanmean(data_err)*3,size=test_spectrum.shape)
   # white noise independent of data points
   white_noise=np.random.normal(np.zeros_like(test_spectrum),data_err,size=test_spectrum.shape)
   if GP==False:
      # add Gaussian noise by using data_err
      test_spectrum_noisy=test_spectrum+random_noise
   else:
      # add correlated noise
      if False: # old version
         Cov = np.empty((n_orders,n_dets), dtype=object) # covariance matrix
         test_spectrum_noisy=np.copy(test_spectrum)
         err_new_array=np.full(shape=test_spectrum.shape,fill_value=np.nan)
         for i in range(n_orders):
            for j in range(n_dets):
                  mask_ij = mask_isfinite[i,j] # only finite pixels
                  if not mask_ij.any(): # skip empty order/detector pairs
                     continue
                  maxval=10**(test_parameters['log_l'])*3 # 3*max value of prior of l
                  Cov[i,j] = CovGauss(err=white_noise[i,j,mask_ij],separation=separation[i,j], 
                                          err_eff=err_eff[i,j],max_separation=maxval)
                  a = 10**(test_parameters.get('log_a'))
                  l = 10**(test_parameters.get('log_l'))
                  Cov[i,j].cov=add_RBF_kernel(Cov[i,j].cov, a, l, separation[i,j], err_eff[i,j])
                  err_new=np.sqrt(Cov[i,j].cov).dot(white_noise[i,j,mask_ij])
                  err_new_array[i,j,mask_ij]=err_new
                  test_spectrum_noisy[i,j,mask_ij]+=err_new
      white_noise=np.random.normal(np.zeros_like(test_spectrum),0.1,size=test_spectrum.shape)
      test_spectrum_noisy=np.copy(test_spectrum)
      err_new_array=np.full(shape=test_spectrum.shape,fill_value=np.nan)
      for i in range(n_orders):
         for j in range(n_dets):
            w=data_wave[i,j]
            err_corr=0.1*np.sin((w-np.nanmean(w))*2)
            err_new=white_noise[i,j]+err_corr
            err_new_array[i,j]=err_new
            test_spectrum_noisy[i,j]+=err_new

   spectrum=np.full(shape=(2048*7*3,3),fill_value=np.nan)
   spectrum[:,0]=data_wave.flatten()
   spectrum[:,1]=test_spectrum_noisy.flatten()
   spectrum[:,2]=data_err.flatten()
   suffix='' if GP==False else '_corr'

   output_dir = pathlib.Path(f'{os.getcwd()}/test{suffix}')
   output_dir.mkdir(parents=True, exist_ok=True)
   np.savetxt(f'test{suffix}/test{suffix}_spectrum.txt',spectrum,delimiter=' ',header='wavelength (nm) flux flux_error')

   if 0:
      wl,fl,err=Target('2M0355').load_spectrum()
      wlm,flm,errm=Target(f'test{suffix}').load_spectrum()
      fig,ax=plt.subplots(1,1,figsize=(9,2),dpi=200)
      ax.plot(wl.flatten(),fl.flatten(),label='2M0355',lw=0.8)
      ax.plot(wlm.flatten(),flm.flatten(),label='testspec',alpha=0.5,lw=0.8)
      ax.legend()
      ax.set_xlabel('Wavelength [nm]')
      fig.tight_layout(h_pad=0)
      fig.savefig(f'test{suffix}/test{suffix}_spectrum.pdf')
      plt.close()