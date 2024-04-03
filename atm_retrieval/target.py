import numpy as np
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

        spectrum=np.load(f"{self.name}.npy")
        wl=spectrum[0] # in nm
        fl=spectrum[1] # unitless
        err=spectrum[2]

        if merge_dets==True: # merge detectors in each order to make shape (orders,wavelength)
            wl=wl.reshape((wl.shape[0],wl.shape[1]*wl.shape[2])) 
            fl=fl.reshape((fl.shape[0],fl.shape[1]*fl.shape[2]))
            err=err.reshape((err.shape[0],err.shape[1]*err.shape[2]))

        if spec_obj==True:
            spec = Spectrum(fl,wl,err)
            return spec
        
        if spec_obj==False:
            return wl,fl,err

