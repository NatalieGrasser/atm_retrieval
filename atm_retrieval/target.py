import numpy as np
from astropy import constants as const
import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    from atm_retrieval.spectrum import Spectrum
elif getpass.getuser() == "natalie": # when testing from my laptop
    from spectrum import Spectrum

class Target:

    def __init__(self,name):
        self.name=name
        if self.name=='2M0355':
            self.ra ="03h55m23.3735910810s"
            self.dec = "+11d33m43.797034332s"
            self.JD=2459885.5   # get JD with https://ssd.jpl.nasa.gov/tools/jdc/#/cd      
        if self.name=='2M1425':
            self.ra="14h25m27.9845344257s"
            self.dec="-36d50m23.248617541s"
            self.JD=2459976.5        

    def load_spectrum(self,spec_obj=False,merge_dets=False):
        spectrum=np.load(f"{self.name}/{self.name}.npy")
        self.wl=spectrum[0] # in nm
        self.fl=spectrum[1] # unitless
        self.err=spectrum[2]
        if merge_dets==True: # merge detectors in each order to make shape (orders,wavelength)
            self.wl=self.wl.reshape((self.wl.shape[0],self.wl.shape[1]*self.wl.shape[2])) 
            self.fl=self.fl.reshape((self.fl.shape[0],self.fl.shape[1]*self.fl.shape[2]))
            self.err=self.err.reshape((self.err.shape[0],self.err.shape[1]*self.err.shape[2]))
        if spec_obj==True: # return Spectrum object
            spec = Spectrum(self.fl,self.wl,self.err)
            return spec
        if spec_obj==False:
            return self.wl,self.fl,self.err
        
    def get_mask_isfinite(self):
        self.n_orders,self.n_dets,self.n_pixels = self.fl.shape # shape (orders,detectors,pixels)
        self.mask_isfinite=np.empty((self.n_orders,self.n_dets,self.n_pixels),dtype=bool)
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                mask_ij = np.isfinite(self.fl[i,j]) # only finite pixels
                self.mask_isfinite[i,j]=mask_ij
        return self.mask_isfinite
    
    def prepare_for_covariance(self):
        self.separation = np.empty((self.n_orders,self.n_dets), dtype=object)
        self.err_eff = np.empty((self.n_orders,self.n_dets), dtype=object)
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                mask_ij = self.mask_isfinite[i,j] # Mask the arrays, on-the-spot is slower
                wave_ij = self.wl[i,j,mask_ij]
                separation_ij = np.abs(wave_ij[None,:]-wave_ij[:,None]) # wavelength separation
                # velocity separation in km/s
                #separation_ij = 2 *const.c.value*1e-3 * np.abs((wave_ij[None,:]-wave_ij[:,None])/(wave_ij[None,:]+wave_ij[:,None])) 
                self.separation[i,j] = separation_ij  
                err_ij = self.err[i,j,mask_ij]  
                #self.err_eff[i,j] = np.sqrt(1/2*(err_ij[None,:]**2 + err_ij[:,None]**2)) # arithmetic mean of squared flux-errors
                self.err_eff[i,j] = np.median(err_ij) # more stable
        return self.separation,self.err_eff

        
        
    

