import numpy as np
from petitRADTRANS import nat_cst as nc
import matplotlib.pyplot as plt
import pathlib
import os

class Target:

    def __init__(self,name):
        self.name=name
        self.n_orders=7
        self.n_dets=3
        self.n_pixels=2048
        if self.name in ['2M0355','testspec']: # testspectrum based on 2M0355
            self.ra ="03h55m23.3735910810s"
            self.dec = "+11d33m43.797034332s"
            self.JD=2459885.5   # get JD with https://ssd.jpl.nasa.gov/tools/jdc/#/cd  
            self.fullname='2MASSJ03552337+1133437'    
            self.standard_star_temp=18700 # lamTau
        if self.name=='2M1425':
            self.ra="14h25m27.9845344257s"
            self.dec="-36d50m23.248617541s"
            self.JD=2459976.5        
            self.fullname='2MASSJ14252798-3650229'
            self.standard_star_temp=10980 # betHya

    def load_spectrum(self):
        self.cwd = os.getcwd()
        file=pathlib.Path(f'{self.cwd}/{self.name}/{self.name}.txt')
        if file.exists():
            file=np.genfromtxt(file,skip_header=1,delimiter=' ')
            self.wl=np.reshape(file[:,0],(self.n_orders,self.n_dets,self.n_pixels))
            self.fl=np.reshape(file[:,1],(self.n_orders,self.n_dets,self.n_pixels))
            self.err=np.reshape(file[:,2],(self.n_orders,self.n_dets,self.n_pixels))
        else:
            # generate useable spectrum from molecfit/output folder
            obj=f'{self.name}/SCIENCE_{self.fullname}_PRIMARY.dat' # target object
            molecfit=f'{self.name}/SCIENCE_{self.fullname}_PRIMARY_molecfit_transm.dat' # molecfit for telluric correction
            self.wl,self.fl,self.err=self.prepare_spectrum(obj,molecfit,temp=self.standard_star_temp,outfile=file)
            self.wl=np.reshape(self.wl,(self.n_orders,self.n_dets,self.n_pixels))
            self.fl=np.reshape(self.fl,(self.n_orders,self.n_dets,self.n_pixels))
            self.err=np.reshape(self.err,(self.n_orders,self.n_dets,self.n_pixels))
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
                self.err_eff[i,j] = np.nanmedian(err_ij) # more stable
        return self.separation,self.err_eff
    
    def load_spec_file(self,file):
        file=np.genfromtxt(file,skip_header=1,delimiter=' ')
        wl=file[:,0]
        fl=file[:,1]
        flerr=file[:,2]
        return wl,fl,flerr
    
    def blackbody(self,wl,temp):
        lamb = wl*1e-7 # wavelength array in cm
        freq = nc.c / lamb # Convert to frequencies
        planck = nc.b(temp, freq) # Calculate Planck function at given temperature (K)
        planck=planck/np.mean(planck)
        return planck
    
    def plot_orders3(self,wl,fl,wl2,fl2,wl3,fl3,label1,label2,label3):
        fig,ax=plt.subplots(self.n_orders,1,figsize=(9,9),dpi=200)
        alph=0.7
        for order in range(self.n_orders):
            ax[order].plot(np.reshape(wl,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           np.reshape(fl,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           lw=0.8,alpha=alph,label=label1)
            ax[order].plot(np.reshape(wl2,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           np.reshape(fl2,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           lw=0.8,alpha=alph,label=label2)
            ax[order].plot(np.reshape(wl3,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           np.reshape(fl3,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           lw=0.8,alpha=alph,label=label3,c='k')
            ax[order].set_xlim(np.min(np.reshape(wl,(self.n_orders,self.n_dets*self.n_pixels))[order]),
                               np.max(np.reshape(wl,(self.n_orders,self.n_dets*self.n_pixels))[order]))
        ax[0].legend(fontsize=8)
        ax[6].set_xlabel('Wavelength [nm]')
        fig.tight_layout(h_pad=0)
        fig.savefig(f'{self.name}/prepare_spectrum.pdf')
        plt.close()

    def prepare_spectrum(self,target,target_tel,temp,outfile=None):
    
        wl,fl0,err=self.load_spec_file(target) # target object
        wlt,flt,continuum=self.load_spec_file(target_tel) # molecfit for telluric correction
        
        fl=np.copy(fl0) # keep for later comparison
        zero_mask=np.where(fl==0)[0] # indices where flux is zero
        fl[zero_mask]=np.nan
        err[zero_mask]=np.nan
        flt[zero_mask]=np.nan
        #fl0[zero_mask]=np.nan

        # mask deepest tellurics: use telluric model because it has a flat baseline
        tel_mask=np.where(flt/np.nanmedian(flt)<0.7)[0]
        fl[tel_mask]=np.nan
        err[tel_mask]=np.nan

        # blackbody of standard star is in continuum, multiply to bring it back
        bb=self.blackbody(wl,temp) 
        fl/=flt/continuum*bb
        err/=flt/continuum*bb

        # mask pixels at beginning and end of each detector
        pm=25 # mask pixels on edge of each detector, plus/minus 20 pixels
        fl=np.reshape(fl,(self.n_orders,self.n_dets,2048))
        err=np.reshape(err,(self.n_orders,self.n_dets,2048))
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                fl[order,det][:pm]=np.nan
                fl[order,det][-pm:]=np.nan
                err[order,det][:pm]=np.nan
                err[order,det][-pm:]=np.nan
        
        # normalize
        fl/=np.nanmedian(fl)
        err/=np.nanmedian(fl)
        flt/=np.nanmedian(flt)

        if self.name=='2M1425': # manually mask weird outlier region
            mean=np.nanmean(fl[2,1])
            std=np.nanstd(fl[2,1])
            bad_pixel=np.where(fl[2,1]>mean+5*std)[0][0] # 890
            fl[2,1][bad_pixel-2:bad_pixel+2]=np.nan

        self.plot_orders3(wl,fl0/np.nanmedian(fl0),wl,flt,wl,fl,'Original','Molecfit','Corrected')
            
        if outfile!=None:
            spectrum=np.full(shape=(self.n_pixels*self.n_orders*self.n_dets,3),fill_value=np.nan)
            spectrum[:,0]=wl.flatten()
            spectrum[:,1]=fl.flatten()
            spectrum[:,2]=err.flatten()
            np.savetxt(outfile,spectrum,delimiter=' ',header='wavelength (nm) flux flux_error')
        
        return wl,fl,err
        
        
    

