import numpy as np
from petitRADTRANS import nat_cst as nc
import matplotlib.pyplot as plt
import pathlib
import os
from numpy.polynomial import polynomial as Poly
from scipy import signal, optimize
from scipy.interpolate import interp1d
import warnings

class Target:

    def __init__(self,name):
        self.name=name
        self.n_orders=7
        self.n_dets=3
        self.n_pixels=2048
        self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                            [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                            [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                            [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                            [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                            [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                            [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])
        
        if self.name in ['2M0355','test','test_corr']: # test spectrum based on 2M0355
            self.ra ="03h55m23.3735910810s"
            self.dec = "+11d33m43.797034332s"
            self.JD=2459885.5   # get JD with https://ssd.jpl.nasa.gov/tools/jdc/#/cd  
            self.fullname='2MASSJ03552337+1133437'    
            self.standard_star_temp=18700 # lamTau
            self.color1='deepskyblue' # color of retrieval output
            self.color2='tab:blue' # color of residuals
            self.color3='lightskyblue' 
            #self.color1='mediumturquoise' # color of retrieval output
            #self.color2='lightseagreen' # color of residuals
            #self.color3='aqua' 
        if self.name=='2M1425':
            self.ra="14h25m27.9845344257s"
            self.dec="-36d50m23.248617541s"
            self.JD=2459976.5        
            self.fullname='2MASSJ14252798-3650229'
            self.standard_star_temp=10980 # betHya
            #self.color1='limegreen' # color of retrieval output
            #self.color2='forestgreen' # color of residuals
            self.color1='lightcoral' # color of retrieval output
            self.color2='indianred' # color of residuals
            self.color3='lightpink'
            #self.color1='mediumpurple' # color of retrieval output
            #self.color2='blueviolet' # color of residuals
            #self.color3='mediumorchid'

    def load_spectrum(self):
        self.cwd = os.getcwd()
        file=pathlib.Path(f'{self.cwd}/{self.name}/{self.name}_spectrum.txt')
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

        if self.name=='2M0355':
            # use corrected wavelength solution, wasn't good for last order-detector
            wlcorr=pathlib.Path(f'{self.cwd}/{self.name}/{self.name}_corr_wl.txt')
            if wlcorr.exists():
                wl=np.genfromtxt(wlcorr,skip_header=1,delimiter=' ')
                self.wl=np.reshape(wl,(self.n_orders,self.n_dets,self.n_pixels))
            else:
                model=np.genfromtxt(f'{self.cwd}/{self.name}/model_spectrum.txt',skip_header=1,delimiter=' ')
                wlm=np.reshape(model[:,0],(self.n_orders,self.n_dets,self.n_pixels))
                flm=np.reshape(model[:,1],(self.n_orders,self.n_dets,self.n_pixels))
                wl_new=self.wlen_solution(self.fl,self.err,self.wl,flm)
                np.savetxt(wlcorr,wl_new.flatten(),delimiter=' ',header='corrected wavelength solution (nm)')
                
                fig = plt.figure(figsize=(9,3),dpi=200)
                plt.plot(self.wl[6,2],self.fl[6,2],label='original')
                plt.plot(wl_new[6,2],self.fl[6,2],label='corrected')
                plt.plot(wlm[6,2],flm[6,2],linestyle='dashed',color='k',alpha=0.7,label='model')
                plt.xlim(np.min(wlm[6,2]),np.max(wlm[6,2]))
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Flux')
                plt.legend()
                fig.tight_layout()
                fig.savefig(f'{self.cwd}/{self.name}/wave_corr_part.pdf')
                plt.close()

                fig = plt.figure(figsize=(9,3),dpi=200)
                for ord in range(7):
                    for det in range(3):
                        plt.plot(self.wl[ord,det],(self.wl[ord,det]-wl_new[ord,det]),color='slateblue')
                plt.xlim(np.min(self.wl)-10,np.max(self.wl)+10)
                plt.xlabel('Original wavelength [nm]')
                plt.ylabel('Original - corrected wavelength [nm]')
                fig.tight_layout()
                fig.savefig(f'{self.cwd}/{self.name}/wavelength_correction.pdf')
                plt.close()

                self.wl=np.reshape(wl_new,(self.n_orders,self.n_dets,self.n_pixels))

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
                           lw=0.8,alpha=1,label=label1,c=self.color1)
            ax[order].plot(np.reshape(wl2,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           np.reshape(fl2,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           lw=0.8,alpha=alph,label=label2,c='yellowgreen')
            ax[order].plot(np.reshape(wl3,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           np.reshape(fl3,(self.n_orders,self.n_dets*self.n_pixels))[order],
                           lw=0.8,alpha=alph,label=label3,c='k')
            ax[order].set_xlim(np.min(np.reshape(wl,(self.n_orders,self.n_dets*self.n_pixels))[order]),
                               np.max(np.reshape(wl,(self.n_orders,self.n_dets*self.n_pixels))[order]))
            ax[order].tick_params(labelsize=8)
        ax[0].legend(fontsize=11,ncol=3,bbox_to_anchor=(0.5,1.4),loc='upper center')
        ax[6].set_xlabel('Wavelength [nm]')
        ax[3].set_ylabel('Normalized Flux')
        fig.tight_layout(h_pad=0.05)
        fig.savefig(f'{self.name}/molecfit_{self.name}.pdf')
        plt.close()

    def prepare_spectrum(self,target,target_tel,temp,outfile=None):
    
        wl,fl0,err=self.load_spec_file(target) # target object
        wlt,flt,continuum=self.load_spec_file(target_tel) # molecfit for telluric correction
        
        fl=np.copy(fl0) # keep for later comparison
        zero_mask=np.where(fl==0)[0] # indices where flux is zero
        fl[zero_mask]=np.nan
        err[zero_mask]=np.nan
        flt[zero_mask]=np.nan
        fl0[zero_mask]=np.nan

        # mask deepest tellurics: use telluric model because it has a flat baseline
        tel_mask=np.where(flt/np.nanmedian(flt)<0.7)[0]
        fl[tel_mask]=np.nan
        err[tel_mask]=np.nan
        fl0_masked=np.copy(fl0)
        fl0_masked[tel_mask]=np.nan

        # blackbody of standard star is in continuum, multiply to bring it back
        bb=self.blackbody(wl,temp) 
        fl=fl/(flt*continuum/bb)
        err=err/(flt*continuum/bb)

        # mask pixels at beginning and end of each detector
        pm=3 # plus/minus mask pixels on edge of each detector -> NECESSARY?
        fl=np.reshape(fl,(self.n_orders,self.n_dets,2048))
        err=np.reshape(err,(self.n_orders,self.n_dets,2048))
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                fl[order,det][:pm]=np.nan
                fl[order,det][-pm:]=np.nan
                err[order,det][:pm]=np.nan
                err[order,det][-pm:]=np.nan

        # has issues, needs more masking...
        pm2=25
        fl[6,2][:pm2]=np.nan
        err[6,2][:pm2]=np.nan
        
        # normalize
        fl/=np.nanmedian(fl)
        err/=np.nanmedian(fl)
        flt/=np.nanmedian(flt)
        fl0/=np.nanmedian(fl0_masked)

        if self.name=='2M1425': # manually mask weird outlier region
            mean=np.nanmean(fl[2,1])
            std=np.nanstd(fl[2,1])
            bad_pixel=np.where(fl[2,1]>mean+5*std)[0][0] # 890
            fl[2,1][bad_pixel-2:bad_pixel+2]=np.nan
            fl[0,:]=np.nan # too many tellurics in entire first order


        self.plot_orders3(wl,fl0,wl,flt,wl,fl,'Uncorrected','Telluric model','Corrected')
            
        if outfile!=None:
            spectrum=np.full(shape=(self.n_pixels*self.n_orders*self.n_dets,3),fill_value=np.nan)
            spectrum[:,0]=wl.flatten()
            spectrum[:,1]=fl.flatten()
            spectrum[:,2]=err.flatten()
            np.savetxt(outfile,spectrum,delimiter=' ',header='wavelength (nm) flux flux_error')
        
        return wl,fl,err

    # functions below from excalibuhr/src/excalibuhr/utils.py
    def func_wlen_optimization(self,poly,*args):
        """
        cost function for optimizing wavelength solutions

        Parameters
        ----------
        poly: array
            polynomial coefficients for correcting the wavelength solution
        args: 
            wave, flux: initial wavelengths and observed flux
            template_interp_func: model spectrum 
        
        Returns
        -------
        correlation: float
            minus correlation between observed and model spectra
        """
        wave, flux, template_interp_func = args
        new_wave = Poly.polyval(wave - np.mean(wave), poly) + wave # apply polynomial coefficients
        template = template_interp_func(new_wave) # interpolate template onto new wavelengths
        correlation = -template.dot(flux) # maximize cross correlation
        return correlation

    def wlen_solution(self,fluxes,errs,w_init,transm_spec,order=2, # use bestfit model as transm_spec
                    p_range=[0.5, 0.05, 0.01],cont_smooth_len=101,debug=False):
        """
        Method for refining wavelength solution using a quadratic 
        polynomial correction Poly(p0, p1, p2). The optimization 
        is achieved by maximizing cross-correlation functions 
        between the spectrum and a telluric transmission model
        NATALIE UPDATE: DOING IT PER ORDER/DET PAIR

        fluxes: array
            flux of observed spectrum in each spectral order
        w_init: array
            initial wavelengths of each spectral order
        p_range: 0th,1st,2nd polynomial coefficient
        cont_smooth_len: int
            the window length used in the high-pass filter to remove 
            the continuum of observed spectrum
        debug : bool
            if True, print the best fit polynomial coefficients.

        Returns
        -------
        wlens: array
            the refined wavelength solution    
        """

        # function to interpolate transmission spectrum
        #template_interp_func = interp1d(w_init,transm_spec,kind='linear')
        wlens = []
        Ncut = 10 # ignore detector-edges 
        minimum_strength=0.0005

        for ord in range(fluxes.shape[0]):
            for det in range(fluxes.shape[1]):
                # function to interpolate transmission spectrum
                template_interp_func = interp1d(w_init[ord,det],transm_spec[ord,det],kind='linear')

                f, f_err, wlen_init = fluxes[ord,det], errs[ord,det], w_init[ord,det]
                f, w, f_err = f[Ncut:-Ncut], wlen_init[Ncut:-Ncut], f_err[Ncut:-Ncut]

                # Remove continuum and nans of spectra
                # continuum estimated by smoothing spectrum with Savitzky-Golay filter
                nans = np.isnan(f)
                continuum = signal.savgol_filter(f[~nans], window_length=cont_smooth_len,polyorder=2, mode='interp')
                f = f[~nans] - continuum
                f, w, f_err = f[Ncut:-Ncut], w[~nans][Ncut:-Ncut], f_err[Ncut:-Ncut]
                bound = [(-p_range[j], p_range[j]) for j in range(order+1)] # 2nd order polynomial -> 3 values

                # Check if there are enough telluric features in this wavelength range
                if np.std(transm_spec) > minimum_strength: 
                    
                    # Use scipy.optimize to find the best-fitting coefficients
                    res = optimize.minimize(self.func_wlen_optimization, args=(w, f, template_interp_func), 
                                            x0=np.zeros(order+1), method='Nelder-Mead', 
                                            tol=1e-8,
                                            bounds=bound) 
                    poly_opt = res.x
                    result = [f'{item:.6f}' for item in poly_opt]
                    if debug:
                        print(f"Order {ord,det} -> Poly(x^0, x^1, x^2): {result}")

                    # if the coefficient hits the prior edge, fitting is unsuccessful
                    # fall back to the 0th oder solution.
                    if np.isclose(np.abs(poly_opt[-1]), p_range[-1]):
                        warnings.warn(f"Fitting of wavelength solution for order {ord,det} is unsuccessful. Only a 0-order offset is applied.")
                        res = optimize.minimize(
                                self.func_wlen_optimization, 
                                args=(w, f, template_interp_func), 
                                x0=[0], method='Nelder-Mead', tol=1e-8, 
                                bounds=[(-p_range[0],+p_range[0])])
                        poly_opt = res.x
                        if debug:
                            print(poly_opt)
                    wlen_cal = wlen_init+Poly.polyval(wlen_init-np.mean(wlen_init),poly_opt)  
                else:
                    warnings.warn(f"Not enough telluric features to correct wavelength for order {ord,det}")
                    wlen_cal = wlen_init
                wlens.append(wlen_cal)

        return np.reshape(np.array(wlens),(self.n_orders,self.n_dets,self.n_pixels))
            
    

