import os
import cloud_cond as cloud_cond
from pRT_model import pRT_spectrum
  
import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, ticker as mticker
from labellines import labelLines
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import warnings
import pathlib
import math
import pandas as pd
from petitRADTRANS import Radtrans
warnings.filterwarnings("ignore", category=UserWarning) 

def scale_between(ymin,ymax,arr):
    scale=(ymax-ymin)/(np.nanmax(arr)-np.nanmin(arr))
    scaled_arr=scale*(arr-np.nanmin(arr))+ymin
    return scaled_arr

def plot_spectrum_inset(retr_obj,inset=True,fs=10,**kwargs):

    wave=retr_obj.data_wave
    flux=retr_obj.data_flux
    err=retr_obj.data_err
    flux_m=retr_obj.model_flux

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(10,2.5),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})
        #fig,ax=plt.subplots(2,1,figsize=(9.5,3),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for order in range(7):
        # add error for scale
        errmean=np.nanmean(err[order]*retr_obj.params_dict['s2_ij'][order].reshape(3,1))
        if np.nansum(flux[order])!=0: # skip empty orders
            #ax[1].fill_between([np.min(wave[order]),np.max(wave[order])],-errmean,errmean,color='k',alpha=0.15)
            ax[1].errorbar(np.min(wave[order])-5, 0, yerr=errmean, ecolor=retr_obj.color1, 
                           elinewidth=1, capsize=2)

        for det in range(3):
            lower=flux[order,det]-err[order,det]*retr_obj.params_dict['s2_ij'][order,det]
            upper=flux[order,det]+err[order,det]*retr_obj.params_dict['s2_ij'][order,det]
            ax[0].plot(wave[order,det],flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            ax[0].fill_between(wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax[0].plot(wave[order,det],flux_m[order,det],lw=0.8,alpha=0.8,c=retr_obj.color1,label='model')
            
            ax[1].plot(wave[order,det],flux[order,det]-flux_m[order,det],lw=0.8,c=retr_obj.color1,label='residuals')
            if order==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        #mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retr_obj.color1, linewidth=2,label='Bestfit')]
                        #Line2D([0], [0], color=retr_obj.color2, linewidth=2,label='Residuals')]
                ax[0].legend(handles=lines,fontsize=fs) # to only have it once
        #ax[1].plot(wave[order].flatten(),np.zeros_like(wave[order].flatten()),lw=0.8,alpha=0.5,c='k')
        ax[1].plot([np.min(wave[order]),np.max(wave[order])],[0,0],lw=0.8,alpha=1,c='k')
        
    ax[0].set_ylabel('Normalized Flux',fontsize=fs)
    ax[1].set_ylabel('Residuals',fontsize=fs)
    ax[0].set_xlim(np.min(wave)-10,np.max(wave)+10)
    ax[0].set_ylim(np.nanmin(np.array([flux,flux_m])),np.nanmax(np.array([flux,flux_m])))
    ax[1].set_xlim(np.min(wave)-10,np.max(wave)+10)
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)

    if inset==True:
        ord=5 
        axins = ax[0].inset_axes([0,-1.3,1,0.75]) # left, bottom, width, height
        for det in range(3):
            lower=flux[ord,det]-err[ord,det]*retr_obj.params_dict['s2_ij'][ord,det]
            upper=flux[ord,det]+err[ord,det]*retr_obj.params_dict['s2_ij'][ord,det]
            axins.fill_between(wave[ord,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            axins.plot(wave[ord,det],flux[ord,det],lw=0.8,c='k')
            axins.plot(wave[ord,det],flux_m[ord,det],lw=0.8,c=retr_obj.color1,alpha=0.8)
        x1, x2 = np.min(wave[ord]),np.max(wave[ord])
        axins.set_xlim(x1, x2)
        box,lines=ax[0].indicate_inset_zoom(axins,edgecolor="black",alpha=0.2,lw=0.8,zorder=1e3)
        axins.set_ylabel('Normalized Flux',fontsize=fs)
        axins.tick_params(labelsize=fs)
        ax[1].set_facecolor('none') # to avoid hiding lines
        ax[0].set_xticks([])
        
        axins2 = axins.inset_axes([0,-0.3,1,0.3])
        for det in range(3):
            axins2.plot(wave[ord,det],flux[ord,det]-flux_m[ord,det],lw=0.8,c=retr_obj.color1)
            axins2.plot(wave[ord,det],np.zeros_like(wave[ord,det]),lw=0.8,alpha=1,c='k')
        axins2.set_xlim(x1, x2)
        axins2.set_xlabel('Wavelength [nm]',fontsize=fs)
        axins2.set_ylabel('Res.',fontsize=fs)
        tick_spacing=1
        axins2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        axins2.tick_params(labelsize=fs)
    else:
        ax[1].set_xlabel('Wavelength [nm]',fontsize=fs) # if no inset

    plt.subplots_adjust(wspace=0, hspace=0)
    if 'ax' not in kwargs:
        name = 'bestfit_spectrum_inset' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_spectrum_inset'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf', bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retr_obj,overplot_species=None,plot_components=False):

    if overplot_species!=None: # overplot species onto residuals
        opacities={}

        for spec in overplot_species:
            from retrieval import Retrieval
            from parameters import Parameters
            from pRT_model import pRT_spectrum
            #opa_orders=[]
            #wave_orders=[]

            # create spectrum containing only selected species
            parameters_species = {} #retr_obj.params_dict.copy()
            for key in retr_obj.parameters.free_params.keys():
                if key in retr_obj.species_names:
                    parameters_species[f"log_{key}"]= -12
                else:
                    parameters_species[key]=retr_obj.params_dict[key]
            if f'log_{spec}' in retr_obj.params_dict.keys():
                parameters_species[f'log_{spec}']=retr_obj.params_dict[f'log_{spec}']
            #else:
                #parameters_species[f'log_{spec}']=-4 # check species that isn't in retrieval
            parameters_species = Parameters({}, parameters_species)
            parameters_species.param_priors['log_l']=[-3,0]
            retrieval_species = Retrieval(target=retr_obj.target,parameters=parameters_species, 
                                          species_names=retr_obj.species_names,
                                        Nlive=retr_obj.Nlive,evtol=retr_obj.evtol,
                                    chemistry='freechem',PT_type=retr_obj.PT_type)            
            if f'log_{spec}' not in retr_obj.params_dict.keys():
                retrieval_species.atmosphere_objects = retrieval_species.get_atmosphere_objects(for_species=spec)
                parameters_species.params[f'log_{spec}']=-4 # check species that isn't in retrieval
                retrieval_species.species = [spec]
            retrieval_species.primary_label=True # to avoid problems
            retrieval_species.model_object=pRT_spectrum(retrieval_species)
            spec_flux=retrieval_species.model_object.make_spectrum()
            for order in range(7):
                for det in range(3):
                    spec_flux[order,det]/=np.median(spec_flux[order,det])
            #print(spec_flux)
            
            #for order in range(7):
            if False:
                    wl_pad=0#7 # wavelength padding because spectrum is not wavelength shifted yet
                    wlmin=np.min(retr_obj.K2166[order])-wl_pad
                    wlmax=np.max(retr_obj.K2166[order])+wl_pad
                    wlen_range=np.array([wlmin,wlmax])*1e-3 # nm to microns
                    atm = Radtrans(line_species=[spec],
                                        rayleigh_species = [],
                                        continuum_opacities = [],
                                        wlen_bords_micron=wlen_range, 
                                        mode='lbl',
                                        lbl_opacity_sampling=3) # take every nth point (=3 in deRegt+2024)
                    
                    T = np.array([1400]).reshape(1)
                    wave_cm, opas = atm.get_opa(T)
                    wave_orders.append(wave_cm*1e7)
                    opa_orders.append(opas[spec].flatten())
                #opacities[spec]=opa_orders
            opacities[spec] = spec_flux
        retr_obj.opacities=opacities

    retrieval=retr_obj
    residuals=(retrieval.data_flux-retrieval.model_flux)
    figsize=(10,13)
    gridspec_kw={'height_ratios':[2,0.9,0.57]*6+[2,0.9]}
    if plot_components==True:
        phi_comp=retrieval.model_object.phi_components
        print('Avg phi star=',np.nanmean(np.nansum(phi_comp.reshape(7*3,8)[:,:-1],axis=1)))
        print('Avg phi BD  =',np.nanmean(phi_comp.reshape(7*3,8)[:,-1]))
        #figsize=(9,15)
        gridspec_kw={'height_ratios':[2,0.3,0.57]*6+[2,0.3]}
    fig,ax=plt.subplots(20,1,figsize=figsize,dpi=200,gridspec_kw=gridspec_kw)
    x=0
    
    for order in range(7): 
        min_array=[np.nanmin([retrieval.data_flux[order],retrieval.model_flux[order]])]
        max_array=[np.nanmax([retrieval.data_flux[order],retrieval.model_flux[order]])]
        ax1=ax[x]
        ax2=ax[x+1]
        
        if x!=18: # last ax cannot be spacer, or xlabel also invisible
            ax3=ax[x+2] #for spacing
        for det in range(3):
            
            ax1.plot(retrieval.data_wave[order,det],retrieval.data_flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            ax1.plot(retrieval.data_wave[order,det],retrieval.model_flux[order,det],lw=0.8,alpha=0.8,c=retr_obj.color1,label='model')
            ax1.set_xlim(np.nanmin(retrieval.data_wave[order])-1,np.nanmax(retrieval.data_wave[order])+1)
            if plot_components==True:
                prim_c='orange'
                sec_c='dodgerblue'
                #prim_flx=phi_comp[order,det][0]*retrieval.primary_flux[order,det]
                #prim_flx= np.sum(phi_comp[order,det][:-1])*retrieval.primary_broadened[order][det]
                #sec_flx = phi_comp[order,det][1]*retrieval.model_object.secondary_flux[order,det]
                prim_flx= retrieval.model_object.primary_broadened[order,det]
                sec_flx = retrieval.model_object.secondary_flux[order,det]
                ax1.plot(retrieval.data_wave[order,det], prim_flx, label='A',lw=0.8, c=prim_c)
                ax1.plot(retrieval.data_wave[order,det], sec_flx,lw=0.8, label='B', c=sec_c)
                #plt.plot(self.data_wave[order][det], np.sum(phi_comp[:-1])*self.primary_broadened[order][det], label='A_broad',c='orange')
                if np.isfinite(phi_comp[order,det].all()) and np.isfinite(prim_flx).any() and np.isfinite(sec_flx).any():
                    min_array.append(np.nanmin([prim_flx,sec_flx]))
                    max_array.append(np.nanmax([prim_flx,sec_flx]))
            ax2.plot(retrieval.data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c=retr_obj.color1,label='residuals')
            ax2.set_xlim(np.nanmin(retrieval.data_wave[order])-1,np.nanmax(retrieval.data_wave[order])+1)

            # add error for scale
            if retrieval.callback_label=='final_':
                lower=retrieval.data_flux[order,det]-retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det]
                upper=retrieval.data_flux[order,det]+retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det]
                ax1.fill_between(retrieval.data_wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
                errmean=np.nanmean(retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det])
                if np.nansum(retrieval.data_flux[order])!=0: # skip empty orders
                    ax2.errorbar(np.min(retrieval.data_wave[order,det])-0.3, 0, yerr=errmean, 
                                 ecolor=retr_obj.color1, elinewidth=1, capsize=2)
            
            if x==0 and det==0:
                ncol=2
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        #mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval.color1, linewidth=2,label='Bestfit')]
                if plot_components==True:
                    lines.append(Line2D([0], [0], color=prim_c,linewidth=2,label='A'))
                    lines.append(Line2D([0], [0], color=sec_c,linewidth=2,label='B'))
                    ncol=4
                leg=ax1.legend(handles=lines,fontsize=12,ncol=ncol,bbox_to_anchor=(0.47,1.4),loc='upper center')
                leg.get_frame().set_linewidth(0.0)

            ax2.plot([np.min(retrieval.data_wave[order,det]),np.max(retrieval.data_wave[order,det])],[0,0],lw=0.8,c='k')

            if overplot_species!=None:
                colors=['b','r','g','y']
                #wl_shifted=retrieval.model_object.wlshift_orders
                for i,species in enumerate(overplot_species):
                    opas=opacities[species]
                    #ymax=np.nanmax(residuals[order])
                    #ymin=np.nanmin(residuals[order])
                    #opa=opas[order]
                    #opa=scale_between(ymin,ymax,opa)
                    ax1.plot(retrieval.data_wave[order,det],opas[order,det],lw=0.8,c=colors[i])

        #min1=np.nanmin(np.array([retrieval.data_flux[order]-retrieval.data_err[order],retrieval.model_flux[order]]))
        #max1=np.nanmax(np.array([retrieval.data_flux[order]+retrieval.data_err[order],retrieval.model_flux[order]]))
        min1=np.nanmin(np.array(min_array))
        max1=np.nanmax(np.array(max_array))
        ax1.set_ylim(min1,max1)
        if np.nansum(residuals[order])!=0:
            ax2.set_ylim(np.nanmin(residuals[order]),np.nanmax(residuals[order]))
        else:# if empty order full of nans
            ax2.set_ylim(-0.1,0.1)
        ax1.tick_params(labelbottom=False)  # don't put tick labels at bottom
        ax1.tick_params(axis="both")
        ax2.tick_params(axis="both")
        ax1.set_ylabel('Normalized Flux')
        ax2.set_ylabel('Res.')
        ax1.tick_params(labelsize=9)
        ax2.tick_params(labelsize=9)
        tick_spacing=1
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        if x!=18:
            ax3.set_visible(False) # invisible for spacing
        x+=3
    ax[19].set_xlabel('Wavelength [nm]')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    if plot_components==False:
        name = 'bestfit_spectrum' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_spectrum'
    else:
        name = 'components' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}components'
    fig.savefig(f'{retr_obj.output_dir}/{name}.pdf')
    plt.close()

def plot_pt(retr_obj,fs=12,**kwargs):

    legend_labels=kwargs.get('legend_labels',None)

    if retr_obj.chemistry in ['equchem','quequchem']:
        C_O = retr_obj.model_object.params['C/O']
        Fe_H = retr_obj.model_object.params['Fe/H']
    if retr_obj.chemistry=='freechem':
        C_O = retr_obj.model_object.CO
        Fe_H = retr_obj.model_object.FeH   

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)
    #cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    #cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    #cs_colors=['gold','goldenrod','peru','sandybrown']
    cloud_species = ['MgSiO3(c)', 'Fe(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)']
    cs_colors=['goldenrod','sandybrown']

    # if pt profile and condensation curve don't intersect, clouds have no effect
    if retr_obj.target.name in ['2M0355','2M1425','test','test_corr']:
        for i,cs in enumerate(cloud_species):
            cs_key = cs[:-3]
            if cs_key == 'KCl':
                cs_key = cs_key.upper()
            P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
            pi=np.where((P_cloud>min(retr_obj.model_object.pressure))&(P_cloud<max(retr_obj.model_object.pressure)))[0]
            ax.plot(T_cloud[pi], P_cloud[pi], lw=1.3, label=cloud_labels[i], ls=':',c=cs_colors[i])
        # https://github.com/cphyc/matplotlib-label-lines
        labelLines(ax.get_lines(),align=False,fontsize=fs*0.8,drop_label=True)
    
    # compare with sonora bobcat T=1400K, logg=4.65 -> 10**(4.65)/100 =  446 m/s²
    #file=np.loadtxt('t1400g562nc_m0.0.dat')
    if retr_obj.target.name in ['test','test_corr']:
        from retrieval import Retrieval
        from parameters import Parameters
        from testspec import test_parameters
        test_par = Parameters({}, test_parameters)
        test_par.param_priors['log_l']=[-3,0]
        test_ret=Retrieval(target=retr_obj.target,parameters=test_par,
                            species_names=retr_obj.species_names, 
                           Nlive=retr_obj.Nlive,evtol=retr_obj.evtol,
                            chemistry='freechem',PT_type='PTgrad')
        test_ret.model_object=pRT_spectrum(test_ret)
        ax.plot(test_ret.model_object.temperature,test_ret.pressure,linestyle='dashdot',c='blueviolet',lw=2) 
        comparison_pt=Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Input')

    elif retr_obj.target.name in ['2M0355','2M1425']:
        file=np.loadtxt('t1600g562nc_m0.0.dat')
        pres=file[:,1] # bar
        temp=file[:,2] # K
        ax.plot(temp,pres,linestyle='dashdot',c='blueviolet',linewidth=2)
        comparison_pt=Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=1600\,$K, log$\,g=4.75$')
    elif retr_obj.target.name in ['ROXs12A','ROXs12B']:
        file=np.loadtxt('t2400g562nc_m0.0.dat')
        pres=file[:,1] # bar
        temp=file[:,2] # K
        ax.plot(temp,pres,linestyle='dashdot',c='blueviolet',linewidth=2)
        comparison_pt=Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=2400\,$K, log$\,g=4.75$')

    if retr_obj.target.name=='2M0355': # compare with Zhang2022 science verification
        PT_Zhang=np.loadtxt(f'{retr_obj.target.name}/2M0355_PT_Zhang2021.dat')
        p_zhang=PT_Zhang[:,0]
        t_zhang=PT_Zhang[:,1]
        ax.plot(t_zhang,p_zhang,linestyle='dashdot',c='cornflowerblue',linewidth=2)

    if 'retr_obj2' in kwargs: # if compare two retrievals, specify object name in legend
        object_label=f'{retr_obj.target.name} retrieval'
        contr_label=f'{retr_obj.target.name} contribution'
    else:
        object_label='$P$-$T$ profile'
        contr_label='Contribution'

    lines=[]
    # plot PT-profile + errors on retrieved temperatures
    def plot_temperature(retr_obj,ax,olabel): 
        if retr_obj.PT_type=='PTknot':
            ax.plot(retr_obj.model_object.temperature,
                retr_obj.model_object.pressure,color=retr_obj.color1,lw=2) 
            medians=[]
            errs=[]
            log_P_knots=retr_obj.model_object.log_P_knots
            for key in ['T4','T3','T2','T1','T0']: # order T4,T3,T2,T1,T0 like log_P_knots
                medians.append(retr_obj.params_dict[key])
                errs.append(retr_obj.params_dict[f'{key}_err'])
            errs=np.array(errs)
            for x in [1,2,3]: # plot 1-3 sigma errors
                lower = CubicSpline(log_P_knots,medians+x*errs[:,0])(np.log10(retr_obj.pressure))
                upper = CubicSpline(log_P_knots,medians+x*errs[:,1])(np.log10(retr_obj.pressure))
                ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color1,alpha=0.15)
            ax.scatter(medians,10**retr_obj.model_object.log_P_knots,color=retr_obj.color1)
            xmin=np.min(lower)-100
            xmax=np.max(upper)+100
            lines.append(Line2D([0],[0],marker='o',color=retr_obj.color1,markerfacecolor=retr_obj.color1,
                    linewidth=2,linestyle='-',label=olabel))

        if retr_obj.PT_type=='PTgrad':
            dlnT_dlnP_knots=[]
            derr=[]
            for i in range(5):
                key=f'dlnT_dlnP_{i}'
                dlnT_dlnP_knots.append(retr_obj.params_dict[key]) # gradient median values
                derr.append(retr_obj.params_dict[f'{key}_err']) # -/+ errors
            derr=np.array(derr) # gradient errors
            T0=retr_obj.params_dict['T0']
            err=retr_obj.params_dict['T0_err']
            temperature=retr_obj.model_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots,T_base=T0)
            ax.plot(temperature,retr_obj.model_object.pressure,color=retr_obj.color1,lw=2) 
            # get 1-2-3 sigma of temp_dist, has shape (samples, n_atm_layers)
            quantiles = np.array([np.percentile(retr_obj.temp_dist[:,i], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1) for i in range(retr_obj.temp_dist.shape[1])])
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,0],quantiles[:,-1],color=retr_obj.color1,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,1],quantiles[:,-2],color=retr_obj.color1,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,2],quantiles[:,-3],color=retr_obj.color1,alpha=0.15)
            xmin=np.min((quantiles[:,0],quantiles[:,-1]))-100
            xmax=np.max((quantiles[:,0],quantiles[:,-1]))+100
            lines.append(Line2D([0], [0], color=retr_obj.color1,
                                linewidth=2,linestyle='-',label=olabel))
        return xmin,xmax

    xmin,xmax=plot_temperature(retr_obj,ax,object_label)
       
    if retr_obj.target.name=='2M0355':
        lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    
    if 'retr_obj2' in kwargs: # compare two retrievals
        retr_obj2=kwargs.get('retr_obj2')
        object_label2=f'{retr_obj2.target.name} retrieval'
        xmin2,xmax2=plot_temperature(retr_obj2,ax,object_label2)
        #lines = lines[:1]+[lines[-1]]+lines[1:-1] # move object2 to second position in legend instead of last
        xmin=np.nanmin([xmin,xmin2])
        xmax=np.nanmax([xmax,xmax2])
        summed_contr2=retr_obj2.summed_contr
        contribution_plot2=summed_contr2/np.max(summed_contr2)*(xmax-xmin)+xmin
        ax.plot(contribution_plot2,retr_obj2.model_object.pressure,linestyle='dashed',
                lw=1.5,alpha=0.8,color=retr_obj2.color2)
        lines.append(Line2D([0], [0], color=retr_obj2.color2, alpha=0.8,linewidth=1.5, 
                            linestyle='--',label=f'{retr_obj2.target.name} contribution'))
        #if retr_obj2.target.name=='2M0355':
            #lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    
    if 'retr_obj3' in kwargs:
        retr_obj3=kwargs.get('retr_obj3')
        object_label3=f'{retr_obj3.target.name} retrieval'
        xmin3,xmax3=plot_temperature(retr_obj3,ax,object_label3)
        xmin=np.nanmin([xmin,xmin2,xmin3])
        xmax=np.nanmax([xmax,xmax2,xmax3])
        summed_contr3=retr_obj3.summed_contr
        contribution_plot3=summed_contr3/np.max(summed_contr3)*(xmax-xmin)+xmin
        ax.plot(contribution_plot3,retr_obj3.model_object.pressure,linestyle='dashed',
                lw=1.5,alpha=0.8,color=retr_obj3.color2)
        lines.append(Line2D([0], [0], color=retr_obj3.color2, alpha=0.8,linewidth=1.5, 
                            linestyle='--',label=f'{retr_obj3.target.name} contribution'))

    summed_contr=retr_obj.summed_contr    
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retr_obj.model_object.pressure,linestyle='dashed',
            lw=1.5,alpha=0.8,color=retr_obj.color2)
    lines.append(Line2D([0], [0], color=retr_obj.color2, alpha=0.8,
                        linewidth=1.5, linestyle='--',label=contr_label))
    lines = lines[:1]+[lines[-1]]+lines[1:-1] # move to second position in legend instead of last
    
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retr_obj.model_object.pressure),
        np.nanmin(retr_obj.model_object.pressure)),xlim=(xmin,xmax))
    
    if legend_labels!=None:
        lines=[]
        retr_objects=[retr_obj,retr_obj2,retr_obj3]
        for r,l in zip(retr_objects,legend_labels):
            lines.append(Line2D([0], [0], color=r.color1,linewidth=2,linestyle='-',label=l))
            #lines.append(Line2D([0], [0], color=r.color2, alpha=0.8,linewidth=1.5, 
                            #linestyle='--',label=f'{l} cont.'))
            
    lines.append(comparison_pt)
    
    ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        name = 'PT_profile' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}PT_profile'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf')
        plt.close()

def get_plotposterior_labels(retr_obj,param_names,posterior_dict):
    param_labels = [] # mathtext  
    plot_posterior=np.empty((len(param_names),len(posterior_dict[list(posterior_dict.keys())[0]][0]))).T
    for i,key in enumerate(param_names):
        param_labels.append(retr_obj.parameters.free_params[key][1]) 
        plot_posterior[:,i]=posterior_dict[key][0].T
    return plot_posterior, param_labels

def cornerplot(retr_obj,getfig=False,figsize=(20,20),fs=12,plot_label='',
            only_abundances=False,only_params=None,not_abundances=False):
    
    posterior=retr_obj.posterior
    param_names = []
    param_labels = [] # mathtext
    if only_abundances==True: # plot only abundances
        plot_label='_abunds'
        for key in retr_obj.species_names:
            param_names.append(f"log_{key}")
            param_labels.append(retr_obj.parameters.free_params[f"log_{key}"][1]) 

    elif only_params is not None: # keys of specified parameters to plot
        param_names = only_params
        for key in only_params:
            param_labels.append(retr_obj.parameters.free_params[key][1]) 

    elif not_abundances==True: # plot all except abundances
        plot_label='_rest'
        set_diff = np.setdiff1d(list(retr_obj.parameters.free_params.keys()),[f'log_{s}' for s in retr_obj.species_names])
        for key in set_diff:
            param_names.append(key)
            param_labels.append(retr_obj.parameters.free_params[key][1]) 
    
    else: # all params: avoid, could be too big
        plot_label='_all'
        param_names= list(retr_obj.parameters.free_params.keys())
        param_labels = list(retr_obj.parameters.param_mathtext.values())

    plot_posterior=np.empty((len(param_names),len(posterior[list(posterior.keys())[0]][0]))).T
    medians = []
    for i,key in enumerate(param_names):
        plot_posterior[:,i]=posterior[key][0].T
        medians.append((retr_obj.get_quantiles(retr_obj.posterior[key][0]))[0])

    fig = plt.figure(figsize=figsize) # fix size to avoid memory issues
    fig = corner.corner(plot_posterior, 
                        labels=param_labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retr_obj.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        hist_kwargs={'density': False,
                                'fill': True,
                                'alpha': 0.5,
                                'edgecolor': 'k',
                                'linewidth': 1.0},
                        fig=fig,
                        quiet=True)
    
    # split title to avoid overlap with plots
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 30: # change 30 to 1 if you want all titles to be split
            title_split = title.split('=')
            titles[i] = title_split[0] + '\n ' + title_split[1]
        fig.axes[i].title.set_text(titles[i])

    #corner.overplot_lines(fig,medians,color=retr_obj.color2,lw=1.3,linestyle='solid') # plot median values of posterior

    # add true values of test spectrum, plotting didn't work bc x-axis range so small, some didn't show up
    if retr_obj.target.name in ['test','test_corr']:
        from testspec import test_parameters,test_mathtext
        compare=np.full(len(param_labels),None) # =None for non-input values of test spectrum
        for key_i in test_parameters.keys():
            label_i=test_mathtext[key_i]
            value_i=test_parameters[key_i]
            if label_i in param_labels:
                i=np.where(param_labels==label_i)[0][0]
                compare[i]=value_i # add only those values that are used in cornerplot, in correct order
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'in: {compare[i]}'
            fig.axes[x].title.set_text(titles[x])
            x+=len(param_labels)+1

        # Adjust tick label font size
        axes = fig.get_axes()
        for ax in axes:
            ax.tick_params(axis="both", labelsize=fs*0.5)  # Tick label font size

    plt.subplots_adjust(wspace=0,hspace=0)

    if getfig==False:
        name= f'cornerplot{plot_label}' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}cornerplot{plot_label}'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
    else:
        ax = np.array(fig.axes)
        return fig, ax

def make_all_plots(retr_obj,only_abundances=False,only_params=None,split_corner=True):
    plot_spectrum_split(retr_obj)
    plot_spectrum_inset(retr_obj)
    plot_pt(retr_obj)
    summary_plot(retr_obj)
    opacity_plot(retr_obj)
    if retr_obj.chemistry=='freechem':
        comp_equ=True # compare with what equchem abundances would be like
        ratios_cornerplot(retr_obj) # already in equchem cornerplot by default
        if split_corner: # split corner plot to avoid massive files
            cornerplot(retr_obj,only_abundances=True)
            cornerplot(retr_obj,not_abundances=True)
        else: # make cornerplot with all parameters, could be huge, avoid this
            cornerplot(retr_obj,only_params=only_params)
    elif retr_obj.chemistry in ['equchem','quequchem']:
        comp_equ=False
        if split_corner: # split corner plot to avoid massive files
            only_params=['rv','vsini','log_g','C/O','Fe/H','log_HF',
                         'log_C12_13_ratio','log_O16_18_ratio','log_O16_17_ratio']
            if retr_obj.chemistry=='quequchem':
                for val in ['log_Pqu_CO_CH4','log_Pqu_NH3','log_Pqu_HCN']:
                    only_params.append(val)
            cornerplot(retr_obj,only_params=only_params,plot_label='1')
            only_params2=list(set(retr_obj.parameters.param_keys)-set(only_params))
            cornerplot(retr_obj,only_params=only_params2,plot_label='2')
        else: # avoid this though
            cornerplot(retr_obj,only_params=only_params)
    VMRs_selected(retr_obj)
    VMRs_all(retr_obj,comp_equ=comp_equ)
    if retr_obj.primary_label==False:
        plot_spectrum_split(retr_obj,plot_components=True)
    
def summary_plot(retr_obj,**kwargs):

    fs=13
    figsize=(17,17)
    if retr_obj.chemistry in ['equchem','quequchem']:
        only_params=['rv','vsini','log_g','T0','C/O','Fe/H','log_HF',
                 'log_C12_13_ratio','log_O16_18_ratio','log_O16_17_ratio']
    if retr_obj.chemistry=='freechem':
        only_params=['rv','vsini','log_g','T0']

        # plot 6 most abundant species
        abunds=[]
        param_names=[]
        species=retr_obj.species_names
        for spec in species:
            abunds.append(retr_obj.params_dict[f'log_{spec}'])
            param_names.append(f"log_{spec}")
        abunds, param_names = zip(*sorted(zip(abunds, param_names)))
        only_params.extend(param_names[-7:][::-1]) # get largest 6
    if 'show_params' in kwargs:
        only_params=kwargs.get('show_params')
        figsize=(15,15)
        fs=9

    fig, ax = cornerplot(retr_obj,getfig=True,only_params=only_params,figsize=figsize,fs=fs)
    l, b, w, h = [0.37,0.84,0.6,0.15] # left, bottom, width, height
    ax_spec = fig.add_axes([l,b,w,h])
    ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
    plot_spectrum_inset(retr_obj,ax=(ax_spec,ax_res),inset=False,fs=fs*1.2)

    l, b, w, h = [0.68,0.47,0.29,0.29] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retr_obj,ax=ax_PT,fs=fs*1.2)
    name = 'summary' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}summary'
    fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()


def double_opacity_plot(retr_obj,retr_obj2,n1=6,n2=7):
    fig,axes=plt.subplots(2,1,figsize=(6,5),dpi=200)
    l1 = opacity_plot(retr_obj,n=n1,ax=axes[0])
    l2= opacity_plot(retr_obj2,n=n2,ax=axes[1],smallrange=True)
    merged = list(set(l1).union(l2))
    lines=[]
    for col,lab in merged:
        lines.append(Line2D([0],[0],color=col,linewidth=2,label=lab))
    legend=axes[0].legend(handles=lines,ncol=(len(lines)//2+len(lines)%2),loc='upper center')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((0, 0, 0, 0))
    name = f'comparison/opacities_both'
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()

def opacity_plot(retr_obj,only_params=None,n=6,smallrange=False,**kwargs):
    Kband=retr_obj.target.K2166

    # plot n most abundant species
    only_params=[]
    abunds=[]
    pRT_names=[]
    labels=[]
    species=retr_obj.species_names

    if retr_obj.chemistry=='freechem':
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        for spec in species:
            abunds.append(retr_obj.params_dict[f"log_{spec}"])
        
    elif retr_obj.chemistry in ['equchem','quequchem']:
        species_info = pd.read_csv(os.path.join('species_info.csv'))
        for spec in species:
            model_object=pRT_spectrum(retr_obj)    
            mass_fractions=model_object.mass_fractions
            MMW=model_object.MMW
            for spec in retr_obj.species:
                mass=species_info.loc[species_info["pRT_name"]==spec]['mass'].values[0]
                abunds.append(np.median(mass_fractions[spec]*MMW/mass)) # take median of abundance
                
    abunds, species = zip(*sorted(zip(abunds, species)))
    only_params=species[-n:][::-1] # get largest 6
    abunds = abunds[-n:][::-1] # get largest 6
    VMRs=[]
    colors=[]
    if retr_obj.chemistry=='freechem':
        for i,par in enumerate(only_params):
            pRT_names.append(species_info.loc[par,'pRT_name'])
            labels.append(species_info.loc[par,'mathtext_name'])
            VMRs.append(10**retr_obj.params_dict[f"log_{only_params[i]}"])
            colors.append(species_info.loc[only_params[i],'color'])
    elif retr_obj.chemistry in ['equchem','quequchem']:
        for i,par in enumerate(only_params):
            pRT_names.append(species_info.loc[species_info["name"]==par]['pRT_name'].values[0])
            labels.append(species_info.loc[species_info["name"]==par]['mathtext_name'].values[0])
            colors.append(species_info.loc[species_info["name"]==par]['color'].values[0])
            VMRs.append(abunds[i])
    
    wlen_range=np.array([np.min(Kband),np.max(Kband)])*1e-3 # nm to microns
    atmosphere = Radtrans(line_species=pRT_names,
                        rayleigh_species = ['H2', 'He'],
                        continuum_opacities = ['H2-H2', 'H2-He'],
                        wlen_bords_micron=wlen_range, 
                        mode='lbl',
                        lbl_opacity_sampling=10)
    
    # use temperature at maximum contribution
    summed_contr = retr_obj.summed_contr
    idx=np.where(summed_contr == np.max(summed_contr))[0][0]
    T = np.array([retr_obj.model_object.temperature[idx]]).reshape(1)
    print("Temperature at maximum contribution = ",T)
    wave_cm, opas = atmosphere.get_opa(T)
    wave_nm = wave_cm*1e7
    if smallrange==False:
        ymin,ymax=5e-8,5e2
    else:
        ymin,ymax=5e-8,1e1

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(6,3),dpi=200)
    lines=[]
    line_props=[]
    for i,m in enumerate(pRT_names):
        #abund=10**retr_obj.params_dict[only_params[i]]
        #col=species_info.loc[f'{only_params[i][4:]}','color']
        #print(names[i],abund)
        spec,=ax.plot(wave_nm,opas[m]*VMRs[i],lw=0.5,c=colors[i])
        lines.append(Line2D([0],[0],color=spec.get_color(),
                        linewidth=2,label=labels[i]))
        line_props.append((spec.get_color(),labels[i]))
        
    for order in range(7):
        for det in range(3):
            ax.fill_betweenx([ymin,ymax],Kband[order,det][0],Kband[order,det][1],color='k',alpha=0.063)
    ax.set_yscale('log')
    ax.set_ylabel('Opacity [cm$^2$/g]')
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim(np.min(retr_obj.target.K2166),np.max(retr_obj.target.K2166))
    ax.set_ylim(ymin,ymax)
    if 'ax' in kwargs:
        return line_props
    else:
        legend=ax.legend(handles=lines,ncol=3,loc='upper center')
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 0, 0))
        legend.get_frame().set_edgecolor((0, 0, 0, 0))
        name = 'opacities' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}opacities'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
    del opas # to avoid memory issues

def CCF_plot(retr_obj,molecule,RVs,CCF_norm,ACF_norm,noiserange=50):
    SNR=CCF_norm[np.where(RVs==0)[0][0]]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5,3.5),dpi=200,gridspec_kw={'height_ratios':[3,1]})
    for ax in (ax1,ax2):
        ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
        ax.set_xlim(np.min(RVs),np.max(RVs))
        ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
        ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
    ax1.plot(RVs,CCF_norm,color=retr_obj.color1,label='CCF')
    ax1.plot(RVs,ACF_norm,color=retr_obj.color1,linestyle='dashed',alpha=0.5,label='ACF')
    ax1.set_ylabel('S/N')
    ax1.legend(loc='upper right')
    if retr_obj.chemistry=='freechem':
        molecule_name=retr_obj.parameters.param_mathtext[f'log_{molecule}'][4:] # remove log_
    elif retr_obj.chemistry in ['equchem','quequchem']:
        if molecule=='13CO':
            molecule_name=r'$^{13}$CO'
        elif molecule=='H2(18)O':
            molecule_name=r'log H$_2^{18}$O'
    molecule_label=f'{molecule_name}\nS/N={np.round(SNR,decimals=1)}'
    #molecule_label=f'{molecule_name}'
    ax1.text(0.05, 0.9, molecule_label,transform=ax1.transAxes,fontsize=14,verticalalignment='top')
    ax2.plot(RVs,CCF_norm-ACF_norm,color=retr_obj.color1)
    ax2.set_ylabel('CCF-ACF')
    ax2.set_xlabel(r'$v_{\rm rad}$ (km/s)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f'{retr_obj.output_dir}/CCF_{molecule}.pdf')
    plt.close()

def compare_retrievals(retr_obj1,retr_obj2,fs=12,with_pt=True,**kwargs): # compare cornerplot+PT of two retrievals

    legend_labels=kwargs.get('legend_labels',None)
    num=2 # number of retrievals
    # can only compare freechem+freechem or freechem+equchem/quequchem(+equchem/quequchem)
    if retr_obj1.chemistry=='freechem' and retr_obj2.chemistry=='freechem':

        if 'show_params' in kwargs:
            only_params=kwargs.get('show_params')
        else:
            only_params=['log_H2O','log_12CO','log_13CO','log_CH4','log_H2S','log_HF','log_H2(18)O','log_NH3']

        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params,retr_obj1.posterior)
        posterior2, _ = get_plotposterior_labels(retr_obj2,only_params,retr_obj2.posterior)

    elif retr_obj1.chemistry=='freechem' and retr_obj2.chemistry in ['equchem','quequchem']:

        only_params=['C/O','C/H','log_12CO/13CO','log_12CO/C17O','log_12CO/C18O','log_H2O/H2(18)O']
        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params,retr_obj1.posterior)

        # add log_O16_18_ratio (last one) to equchem again, bc freechem has C18O and H218O ratios
        only_params=['C/O','Fe/H','log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_O16_18_ratio']
        posterior2, labels = get_plotposterior_labels(retr_obj2,only_params,retr_obj2.posterior)

        if 'retr_obj3' in kwargs: # quequchem
            num=3
            retr_obj3=kwargs.get('retr_obj3')
            posterior3, labels = get_plotposterior_labels(retr_obj3,only_params,retr_obj3.posterior)

    figsize=13
    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues
    
    def plot_corner(posterior,retr_obj,labels,fig,getfig=False):
        ranges = [(np.min([np.min(posterior1[:, i]),np.min(posterior2[:, i])]), 
                   np.max([np.max(posterior1[:, i]),np.max(posterior2[:, i])])) for i in range(posterior1.shape[1])]
        #ranges[3] = (-11,-4.7) # increase range for CH4 to make better visible
        #ranges[4] = (-11,-3.5) # increase range for H2S to make better visible
        #ranges[6] = (-12,-4.7) # increase range for H218O to make better visible
        fig = corner.corner(posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize': fs},
                        label_kwargs={'fontsize': fs*0.8},
                        color=retr_obj.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        plot_contours=True,
                        hist_kwargs={'density': False,
                                    'fill': True,
                                    'alpha': 0.5,
                                    'edgecolor': 'k',
                                    'linewidth': 1.0},
                        fig=fig,
                        quiet=True,
                        range=ranges)
        
        titles = [axi.title.get_text() for axi in fig.axes]
        if getfig:
            return fig,titles
        else:
            return titles
        
    fig,titles1=plot_corner(posterior1,retr_obj1,labels,fig,getfig=True)
    titles2=plot_corner(posterior2,retr_obj2,labels,fig)
    enum=[0,1]
    titles_list=[titles1,titles2]
    colors_list=[retr_obj1.color1,retr_obj2.color1]

    if 'retr_obj3' in kwargs:
        titles3=plot_corner(posterior3,retr_obj3,labels,fig)
        enum=[0,1,2]
        titles_list.append(titles3)
        colors_list.append(retr_obj3.color1)

    for i, axi in enumerate(fig.axes):
        fig.axes[i].title.set_visible(False) # remove original titles
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)
        
    for run,titles_list,color in zip(enum,titles_list,colors_list):
        # add new titles
        for j, title in enumerate(titles_list):
            if title == '':
                continue
            
            # first only the name of the parameter
            s = title.split('=')
            if len(enum)==2:
                y_title=1.45
                y=y_title-(0.2*(run+1))
            elif len(enum)==3:
                y_title=1.47
                y=y_title-0.12-(0.13*(run+0.2))
            if run == 0: # first retrieval, add parameter name
                fig.axes[j].text(0.5, y_title, s[0], fontsize=fs,
                                ha='center', va='bottom',
                                transform=fig.axes[j].transAxes,
                                color='k',
                                weight='normal')
            # add parameter value with custom color and spacing
            fig.axes[j].text(0.5, y, s[1], fontsize=fs,
                            ha='center', va='bottom',
                            transform=fig.axes[j].transAxes,
                            color=color,
                            weight='normal')

    plt.subplots_adjust(wspace=0, hspace=0)

    if with_pt==True:
        l, b, w, h = [0.58,0.65,0.39,0.39] # left, bottom, width, height
        ax_PT = fig.add_axes([l,b,w,h])
        if 'retr_obj3' not in kwargs:
            plot_pt(retr_obj1,retr_obj2=retr_obj2,ax=ax_PT)
        else:
            plot_pt(retr_obj1,retr_obj2=retr_obj2,
                    ax=ax_PT,retr_obj3=retr_obj3,legend_labels=legend_labels)

    comparison_dir=pathlib.Path(f'{retr_obj1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(f'{comparison_dir}/cornerplot_{num}.pdf',bbox_inches="tight",dpi=200)
    plt.close()

def compare_two_CCFs(retr_obj1,retr_obj2,molecules,noiserange=100):

    RVs=np.arange(-500,500,1) # km/s
    comparison_dir=pathlib.Path(f'{retr_obj1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for i,molecule in enumerate(molecules):

        CCF_norm1=retr_obj1.CCF_list[i]
        ACF_norm1=retr_obj1.ACF_list[i]
        SNR1=CCF_norm1[np.where(RVs==0)[0][0]]
        CCF_norm2=retr_obj2.CCF_list[i]
        ACF_norm2=retr_obj2.ACF_list[i]
        SNR2=CCF_norm2[np.where(RVs==0)[0][0]]

        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5,3.5),dpi=200,gridspec_kw={'height_ratios':[3,1]})
        for ax in (ax1,ax2):
            ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
            ax.set_xlim(np.min(RVs),np.max(RVs))
            ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
            ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
        ax1.plot(RVs,CCF_norm1,color=retr_obj1.color1,label=f'{retr_obj1.target.name}')
        ax1.plot(RVs,ACF_norm1,color=retr_obj1.color1,linestyle='dashed',alpha=0.5)

        ax1.plot(RVs,CCF_norm2,color=retr_obj2.color1,label=f'{retr_obj2.target.name}')
        ax1.plot(RVs,ACF_norm2,color=retr_obj2.color1,linestyle='dashed',alpha=0.5)

        lines = [Line2D([0], [0], color=retr_obj1.color1,linewidth=2,label=f'{retr_obj1.target.name}'),
                 Line2D([0], [0], color=retr_obj2.color1,linewidth=2,label=f'{retr_obj2.target.name}'),
                 Line2D([0], [0], color='k',linewidth=2,alpha=0.5,label='CCF'),
                 Line2D([0], [0], color='k',linestyle='--',linewidth=2,alpha=0.2,label='ACF')]
        ax1.legend(handles=lines,fontsize=9,loc='upper right')
        ax1.set_ylabel('S/N')

        molecule_label=str(retr_obj1.parameters.param_mathtext[f'log_{molecule}'][4:]) # remove log_
        ax1.text(0.05, 0.9, molecule_label,transform=ax1.transAxes,fontsize=14,verticalalignment='top')

        ax2.plot(RVs,CCF_norm1-ACF_norm1,color=retr_obj1.color1)
        ax2.plot(RVs,CCF_norm2-ACF_norm2,color=retr_obj2.color1)
        ax2.set_ylabel('CCF-ACF')
        ax2.set_xlabel(r'$v_{\rm rad}$ (km/s)')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        filename=f'CCF_{molecule}_2.pdf'
        fig.savefig(f'{comparison_dir}/{filename}',bbox_inches="tight",dpi=200)
        plt.close()

def ratios_cornerplot(retr_obj,fs=10,**kwargs):

    if retr_obj.chemistry in ['equchem','quequchem']:
        labels=['C/O','[Fe/H]',r'log $^{12}$C/$^{13}$C',r'log $^{16}$O/$^{17}$O',r'log $^{16}$O/$^{18}$O']
    elif retr_obj.chemistry=='freechem':
        labels=['C/O','[C/H]',r'log $^{12}$CO/$^{13}$CO',r'log $^{12}$CO/C$^{17}$O',
                r'log $^{12}$CO/C$^{18}$O',r'log H$_2^{16}$O/H$_2^{18}$O']
    
    fig = plt.figure(figsize=(8,8)) # fix size to avoid memory issues
    fig = corner.corner(retr_obj.ratios_posterior,
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retr_obj.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        hist_kwargs={'density': False,
                                'fill': True,
                                'alpha': 0.5,
                                'edgecolor': 'k',
                                'linewidth': 1.0},
                        fig=fig,
                        quiet=True)
    titles = [axi.title.get_text() for axi in fig.axes]
    
    if 'retr_obj2' in kwargs: # compare two retrievals
        retr_obj2=kwargs.get('retr_obj2')
        corner.corner(retr_obj2.ratios_posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retr_obj2.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        hist_kwargs={'density': False,
                                'fill': True,
                                'alpha': 0.5,
                                'edgecolor': 'k',
                                'linewidth': 1.0},
                        fig=fig,
                        quiet=True)
        titles2 = [axi.title.get_text() for axi in fig.axes]

        for run,titles_list,color in zip([0,1],[titles,titles2],[retr_obj.color1,retr_obj2.color1]):
            # add new titles
            for j, title in enumerate(titles_list):
                if title == '':
                    continue
                
                # first only the name of the parameter
                s = title.split('=')
                if run == 0: # first retrieval, add parameter name
                    fig.axes[j].text(0.5, 1.45, s[0], fontsize=fs,
                                    ha='center', va='bottom',
                                    transform=fig.axes[j].transAxes,
                                    color='k',
                                    weight='normal')
                # add parameter value with custom color and spacing
                fig.axes[j].text(0.5, 1.45-(0.2*(run+1)), s[1], fontsize=fs,
                                ha='center', va='bottom',
                                transform=fig.axes[j].transAxes,
                                color=color,
                                weight='normal')
                
        for i, axi in enumerate(fig.axes):
            fig.axes[i].title.set_visible(False) # remove original titles   
        comparison_dir=pathlib.Path(f'{retr_obj.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        filename=f'{comparison_dir}/ratios_2.pdf'

    else:
        for i, title in enumerate(titles):
            if len(title) > 1: # change 30 to 1 if you want all titles to be split
                title_split = title.split('=')
                titles[i] = title_split[0] + '\n ' + title_split[1]
            fig.axes[i].title.set_text(titles[i])
        name1=f'{retr_obj.output_dir}/ratios.pdf'
        name2=f'{retr_obj.output_dir}/{retr_obj.callback_label}ratios.pdf'
        filename = name1 if retr_obj.callback_label=='final_' else name2

    for i, axi in enumerate(fig.axes):
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)

    plt.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(filename,bbox_inches="tight",dpi=200)
    plt.close()

def VMRs_all(retr_obj,fs=10,comp_equ=False,**kwargs):

    #prefix=retr_obj.callback_label if retr_obj.callback_label=='final_' else ''
    prefix=''
    suffix=''
    output_dir=retr_obj.output_dir
    fig,ax=plt.subplots(1,1,figsize=(6,4),dpi=200)
    species_info = pd.read_csv(os.path.join('species_info.csv'))
    molecules=retr_obj.species_names
    alpha=0.6 if 'retr_obj2' in kwargs or comp_equ==True else 1
    legend_labels=0
    xmin,xmax=1e-11,1.3
    chemleg=[] # legend for chemistry

    def plot_VMRs(retr_obj,ax,ax2):
        if retr_obj.chemistry=='freechem':
            linestyle='dashed'
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=0.5,label='Free'))
        elif retr_obj.chemistry=='equchem':
            linestyle='solid'
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=0.2,label='Equ'))
        elif retr_obj.chemistry=='quequchem':
            linestyle='dotted'
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=0.2,label='Quench'))
        mass_fractions=retr_obj.model_object.mass_fractions
        MMW=retr_obj.model_object.MMW
        params=retr_obj.parameters.params
        for species in molecules:
            name=species_info.loc[species_info["name"]==species]['pRT_name'].values[0]
            mass=species_info.loc[species_info["name"]==species]['mass'].values[0]
            label=species_info.loc[species_info["name"]==species]['mathtext_name'].values[0]
            color=species_info.loc[species_info["name"]==species]['color'].values[0]
            if retr_obj.chemistry=='freechem':
                VMR=mass_fractions[name]*MMW/mass
                label=label if legend_labels==0 else '_nolegend_' 
                ax.plot(VMR,pressure,label=label,linestyle=linestyle,c=color)
            elif retr_obj.chemistry in ['equchem','quequchem']:
                if species not in ['H2(18)O','13CO','C18O','C17O']:
                    VMR=mass_fractions[name]*MMW/mass
                    label=label if legend_labels==0 else '_nolegend_'
                    ax.plot(VMR,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                else:  
                    H2Oname=species_info.loc[species_info["name"]=='H2O']['pRT_name'].values[0]
                    VMR_H2O=mass_fractions[H2Oname]*MMW/species_info.loc[species_info["name"]=='H2O']['mass'].values[0]
                    COname=species_info.loc[species_info["name"]=='12CO']['pRT_name'].values[0]
                    VMR_12CO=mass_fractions[COname]*MMW/species_info.loc[species_info["name"]=='12CO']['mass'].values[0]
                    if species=='H2(18)O':                        
                        VMR_H218O=10**(-params.get('log_O16_18_ratio',-12))*VMR_H2O
                        label=r'H$_2^{18}$O' if legend_labels==0 else '_nolegend_'
                        ax.plot(VMR_H218O,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                    elif species=='13CO':
                        VMR_13CO=10**(-params.get('log_C12_13_ratio',-12))*VMR_12CO
                        label=r'$^{13}$CO' if legend_labels==0 else '_nolegend_'
                        ax.plot(VMR_13CO,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                    elif species=='C18O':
                        VMR_C18O=10**(-params.get('log_O16_18_ratio',-12))*VMR_12CO
                        label=r'C$^{18}$O' if legend_labels==0 else '_nolegend_'
                        ax.plot(VMR_C18O,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                    elif species=='C17O':
                        VMR_C17O=10**(-params.get('log_O16_17_ratio',-12))*VMR_12CO
                        label=r'C$^{17}$O' if legend_labels==0 else '_nolegend_'
                        ax.plot(VMR_C17O,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)

        summed_contr=retr_obj.summed_contr
        contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
        ax2.plot(contribution_plot,np.log10(retr_obj.model_object.pressure)[::-1],
                    lw=1,alpha=alpha*0.5,color=retr_obj.color1,linestyle=linestyle)
        ax2.set_xlim(np.min(contribution_plot),np.max(contribution_plot))

    pressure=retr_obj.model_object.pressure
    ax2 = ax.inset_axes([0,0,1,1]) # [x0, y0, width, height] , for emission contribution

    plot_VMRs(retr_obj,ax=ax,ax2=ax2)
    legend_labels=1 # only make legend labels once 

    # compare freechem VMRs to equilibrium chemistry with other retrieved params remainig equal
    if comp_equ==True:
        from retrieval import Retrieval
        from parameters import Parameters
        parameters_equ = retr_obj.params_dict
        parameters_equ.update({'C/O': retr_obj.params_dict['C/O'],
                        'Fe/H': retr_obj.params_dict['C/H'],
                        'log_C12_13_ratio': retr_obj.params_dict['log_12CO/13CO'],
                        'log_O16_18_ratio': retr_obj.params_dict['log_H2O/H2(18)O'],
                        'log_O16_17_ratio': retr_obj.params_dict['log_12CO/C17O']})
        parameters_equ = Parameters({}, parameters_equ)
        parameters_equ.param_priors['log_l']=[-3,0]
        retrieval_equ = Retrieval(target=retr_obj.target,parameters=parameters_equ, 
                                  species_names=retr_obj.species_names,
                                  Nlive=retr_obj.Nlive,evtol=retr_obj.evtol,
                                chemistry='equchem',PT_type=retr_obj.PT_type)
        retrieval_equ.primary_label=True # to avoid problems
        retrieval_equ.model_object=pRT_spectrum(retrieval_equ,contribution=True)
        retrieval_equ.model_object.make_spectrum() # to get emission contribution
        retrieval_equ.summed_contr=np.nanmean(retrieval_equ.model_object.contr_em_orders,axis=0) # average over all orders
        plot_VMRs(retrieval_equ,ax=ax,ax2=ax2)

    if 'retr_obj2' in kwargs: # compare two retrievals
        prefix=''
        suffix='_2'
        retr_obj2=kwargs.get('retr_obj2')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retr_obj2,ax=ax,ax2=ax2)
        comparison_dir=pathlib.Path(f'{retr_obj.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        output_dir=comparison_dir

    if 'retr_obj3' in kwargs: # compare three retrievals
        suffix='_3'
        retr_obj3=kwargs.get('retr_obj3')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retr_obj3,ax=ax,ax2=ax2)

    leg=ax.legend(fontsize=fs*0.8,loc='lower left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    for line in leg.get_lines():
        line.set_linestyle('-')
    ax.add_artist(leg)
    if comp_equ==True:
        leg2=ax.legend(handles=chemleg,fontsize=fs*0.8,loc='upper left')
        ax.add_artist(leg2)
    
    ax2.axis('off')
    ax2.set_facecolor('none')
    ax.set(xlabel='VMR', ylabel='Pressure [bar]',yscale='log',xscale='log',
        ylim=(np.max(pressure),np.min(pressure)),xlim=(xmin,xmax))   
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('VMR', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)
    fig.tight_layout()
    callback_suffix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{output_dir}/{prefix}VMRs_all{callback_suffix}{suffix}.pdf')
    plt.close()

def CCFs_molecules(retr_obj1,retr_obj2,molecules,noiserange=100):

    RVs=np.arange(-500,500,1) # km/s
    comparison_dir=pathlib.Path(f'{retr_obj1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig,ax=plt.subplots(len(molecules),1,figsize=(5,len(molecules)*1.5),dpi=200)
    for i in range(len(molecules)):
        ax[i].axvspan(-noiserange,noiserange,color='k',alpha=0.05)
        ax[i].set_xlim(np.min(RVs),np.max(RVs))
        ax[i].axvline(x=0,color='k',lw=0.6,alpha=0.3)
        ax[i].axhline(y=0,color='k',lw=0.6,alpha=0.3)

    for i,molecule in enumerate(molecules):

        CCF_norm1=retr_obj1.CCF_list[i]
        ACF_norm1=retr_obj1.ACF_list[i]
        SNR1=CCF_norm1[np.where(RVs==0)[0][0]]
        CCF_norm2=retr_obj2.CCF_list[i]
        ACF_norm2=retr_obj2.ACF_list[i]
        SNR2=CCF_norm2[np.where(RVs==0)[0][0]]

        ax[i].plot(RVs,CCF_norm1,color=retr_obj1.color1,label=f'{retr_obj1.target.name}')
        ax[i].plot(RVs,ACF_norm1,color=retr_obj1.color1,linestyle='dashed',alpha=0.5)
        ax[i].plot(RVs,CCF_norm2,color=retr_obj2.color1,label=f'{retr_obj2.target.name}')
        ax[i].plot(RVs,ACF_norm2,color=retr_obj2.color1,linestyle='dashed',alpha=0.5)
        ax[i].set_ylabel('S/N')
        molecule_label=str(retr_obj1.parameters.param_mathtext[f'log_{molecule}'][4:]) # remove log_
        ax[i].text(0.05, 0.9, molecule_label,transform=ax[i].transAxes,fontsize=14,verticalalignment='top')

    lines = [Line2D([0], [0], color=retr_obj1.color1,linewidth=2,label=f'{retr_obj1.target.name}'),
                 Line2D([0], [0], color=retr_obj2.color1,linewidth=2,label=f'{retr_obj2.target.name}'),
                 Line2D([0], [0], color='k',linewidth=2,alpha=0.5,label='CCF'),
                 Line2D([0], [0], color='k',linestyle='--',linewidth=2,alpha=0.2,label='ACF')]
    ax[0].legend(handles=lines,fontsize=9,loc='upper right')
    ax[len(molecules)-1].set_xlabel(r'$v_{\rm rad}$ (km/s)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    filename=f'CCFs_{len(molecules)}.pdf'
    fig.savefig(f'{comparison_dir}/{filename}',bbox_inches="tight",dpi=200)
    plt.close()

def compare_metall_logg(retr_obj1,retr_obj2,fs=12,only_params=['log_g'],**kwargs):
    
    labels=list(retr_obj1.parameters.param_mathtext.values())
    indices=[]
    for key in only_params:
        idx=list(retr_obj1.parameters.params).index(key)
        indices.append(idx)
    posterior1=np.array([retr_obj1.posterior[:,i] for i in indices]).T
    posterior2=np.array([retr_obj2.posterior[:,i] for i in indices]).T
    labels=np.array([labels[i] for i in indices])

    # add C/O and C/H
    total_posterior1=np.hstack([posterior1,retr_obj1.ratios_posterior[:,0:1],
                                retr_obj1.ratios_posterior[:,1:2],
                                retr_obj1.ratios_posterior[:,2:3],
                                retr_obj1.ratios_posterior[:,5:6]]) 
    total_posterior2=np.hstack([posterior2,retr_obj2.ratios_posterior[:,0:1],
                                retr_obj2.ratios_posterior[:,1:2],
                                retr_obj2.ratios_posterior[:,2:3],
                                retr_obj2.ratios_posterior[:,5:6]])
    labels=np.hstack([labels,'C/O','[C/H]',r'log $^{12}$CO/$^{13}$CO',r'log H$_2^{16}$O/H$_2^{18}$O'])
    
    fig = plt.figure(figsize=(8,8)) # fix size to avoid memory issues

    figsize=8
    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues

    def plot_corner(posterior,retr_obj,labels,fig,getfig=False):
        ranges = [(np.min([np.min(total_posterior1[:, i]),np.min(total_posterior2[:, i])]), 
                   np.max([np.max(total_posterior1[:, i]),np.max(total_posterior2[:, i])])) for i in range(total_posterior2.shape[1])]
        ranges[0] = (4.6,5.02) # increase range for logg to make better visible
        ranges[4] = (2,8)
        fig = corner.corner(posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize': fs},
                        label_kwargs={'fontsize': fs*0.8},
                        color=retr_obj.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        plot_contours=True,
                        hist_kwargs={'density': False,
                                    'fill': True,
                                    'alpha': 0.5,
                                    'edgecolor': 'k',
                                    'linewidth': 1.0},
                        fig=fig,
                        quiet=True,
                        range=ranges)
        
        titles = [axi.title.get_text() for axi in fig.axes]
        if getfig:
            return fig,titles
        else:
            return titles
        
    fig,titles1=plot_corner(total_posterior1,retr_obj1,labels,fig,getfig=True)
    titles2=plot_corner(total_posterior2,retr_obj2,labels,fig)
    enum=[0,1]
    titles_list=[titles1,titles2]
    colors_list=[retr_obj1.color1,retr_obj2.color1]

    for i, axi in enumerate(fig.axes):
        fig.axes[i].title.set_visible(False) # remove original titles
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)
        
    for run,titles_list,color in zip(enum,titles_list,colors_list):
        # add new titles
        for j, title in enumerate(titles_list):
            if title == '':
                continue
            
            # first only the name of the parameter
            s = title.split('=')
            if len(enum)==2:
                y_title=1.45
                y=y_title-(0.2*(run+1))
            elif len(enum)==3:
                y_title=1.47
                y=y_title-0.12-(0.13*(run+0.2))
            if run == 0: # first retrieval, add parameter name
                fig.axes[j].text(0.5, y_title, s[0], fontsize=fs,
                                ha='center', va='bottom',
                                transform=fig.axes[j].transAxes,
                                color='k',
                                weight='normal')
            # add parameter value with custom color and spacing
            fig.axes[j].text(0.5, y, s[1], fontsize=fs,
                            ha='center', va='bottom',
                            transform=fig.axes[j].transAxes,
                            color=color,
                            weight='normal')

    plt.subplots_adjust(wspace=0, hspace=0)
    comparison_dir=pathlib.Path(f'{retr_obj1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{comparison_dir}/cornerplot_ratios_logg.pdf',bbox_inches="tight",dpi=200)
    plt.close()

def VMRs_selected(retr_obj,fs=10,n=6,molecules=None,comp_equ=False,**kwargs):

    #prefix=retr_obj.callback_label if retr_obj.callback_label=='final_' else ''
    prefix=''
    suffix=''
    output_dir=retr_obj.output_dir
    fig,ax=plt.subplots(1,1,figsize=(5,3.5),dpi=200)
    species_info = pd.read_csv(os.path.join('species_info.csv'))
    alpha=0.6 if 'retr_obj2' in kwargs or comp_equ==True else 1
    legend_labels=0
    xmin,xmax=1e-10,1e-2
    chemleg=[] # legend for chemistry
    pressure=retr_obj.model_object.pressure

    # plot n most abundant species
    if molecules==None:
        abunds=[]
        species=retr_obj.species_names
        if retr_obj.chemistry=='freechem':
            for i,spec in enumerate(species):
                abunds.append(retr_obj.params_dict[f"log_{spec}"])
        elif retr_obj.chemistry in ['equchem','quequchem']:
            for spec in species:
                model_object=pRT_spectrum(retr_obj)    
                mass_fractions=model_object.mass_fractions
                MMW=model_object.MMW
                for spec in retr_obj.species:
                    mass=species_info.loc[species_info["pRT_name"]==spec]['mass'].values[0]
                    abunds.append(np.median(mass_fractions[spec]*MMW/mass)) # take median of abundance            
        abunds, species = zip(*sorted(zip(abunds, species)))
        molecules=species[-n:][::-1] # get n largest

    def plot_VMRs(retr_obj,ax,ax2):
        
        if retr_obj.chemistry=='freechem' and 'retr_obj2' not in kwargs:
            linestyle='dashed'
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=0.7,label='Free'))
        elif retr_obj.chemistry=='freechem' and 'retr_obj2' in kwargs:    
            chemleg.append(Line2D([0], [0], marker='o',color='k',markerfacecolor='k',linewidth=2,alpha=0.7,label='Free'))
        elif retr_obj.chemistry=='equchem':
            linestyle='solid'
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=0.3,label='Equ'))
        elif retr_obj.chemistry=='quequchem':
            linestyle='dotted'
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=0.3,label='Quench'))

        if retr_obj.chemistry=='freechem':
            contribution_plot=retr_obj.summed_contr/np.max(retr_obj.summed_contr)*(xmax-xmin)+xmin
            ax2.plot(contribution_plot,retr_obj.model_object.pressure[::-1],
                    lw=1,alpha=0.3,color=retr_obj.color1,linestyle='dashdot')
            ax2.set_xlim(np.min(contribution_plot),np.max(contribution_plot))
            ax2.set_ylim(np.min(pressure),np.max(pressure))
            contr_max=pressure[np.where(retr_obj.summed_contr==np.max(retr_obj.summed_contr))[0]]
            ax2.set_yscale('log')

        for species in molecules:
            color=species_info.loc[species_info["name"]==species]['color'].values[0]
            label=species_info.loc[species_info["name"]==species]['mathtext_name'].values[0]
            if retr_obj.chemistry=='freechem':
                label=label if legend_labels==0 else '_nolegend_' 
                VMR=10**retr_obj.params_dict[f'log_{species}']
                sm3,sm2,sm1,median,sp1,sp2,sp3 = 10**np.array(np.percentile(retr_obj.posterior[f'log_{species}'][0],[0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1))
                if 'retr_obj2' not in kwargs:
                    ax.plot(np.ones_like(pressure)*VMR,pressure,label=label,linestyle=linestyle,c=color)
                    ax.fill_betweenx(pressure,sm2,sp2,color=color,alpha=0.1) # 95% confidence interval
                else: # plot only as point to avoid cluttering
                    ax.scatter(VMR,contr_max, color=color,s=13)
                    ax.plot([sm2,sp2],[contr_max,contr_max], color=color,lw=1.5)
            elif retr_obj.chemistry in ['equchem','quequchem']:
                label=label if legend_labels==0 else '_nolegend_'
                if comp_equ==False:
                    sm3,sm2,sm1,median,sp1,sp2,sp3=np.percentile(retr_obj.VMR_dict[species], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=0)
                    ax.plot(median,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                    if retr_obj.chemistry=='equchem':
                        ax.fill_betweenx(pressure,sm2,sp2,color=color,alpha=0.1) # 95% confidence interval
                    elif retr_obj.chemistry=='quequchem':
                        ax.fill_betweenx(pressure,sm2,sp2,color=color,alpha=0.05,hatch='x') # 95% confidence interval
                else:
                    ax.plot(retr_obj.model_object.VMRs[species],pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)

    
    ax2 = ax.inset_axes([0,0,1,1]) # [x0, y0, width, height] , for emission contribution

    plot_VMRs(retr_obj,ax=ax,ax2=ax2)
    legend_labels=1 if 'retr_obj2' not in kwargs else 0 # only make legend labels once 

    # compare freechem VMRs to equilibrium chemistry with other retrieved params remaining equal
    if comp_equ==True:
        from retrieval import Retrieval
        from parameters import Parameters

        parameters_equ = retr_obj.params_dict
        parameters_equ.update({'C/O': retr_obj.params_dict['C/O'],
                        'Fe/H': retr_obj.params_dict['C/H'],
                        'log_C12_13_ratio': retr_obj.params_dict['log_12CO/13CO'],
                        'log_O16_18_ratio': retr_obj.params_dict['log_H2O/H2(18)O'],
                        'log_O16_17_ratio': retr_obj.params_dict['log_12CO/C17O']})
        parameters_equ = Parameters({}, parameters_equ)
        parameters_equ.param_priors['log_l']=[-3,0]
        retrieval_equ = Retrieval(target=retr_obj.target,parameters=parameters_equ, 
                                  species_names=retr_obj.species_names,
                                  Nlive=retr_obj.Nlive,evtol=retr_obj.evtol,
                                chemistry='equchem',PT_type=retr_obj.PT_type)
        retrieval_equ.model_object=pRT_spectrum(retrieval_equ)
        plot_VMRs(retrieval_equ,ax=ax,ax2=ax2)

    if 'retr_obj2' in kwargs: # compare two retrievals
        prefix=''
        suffix='_2'
        retr_obj2=kwargs.get('retr_obj2')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retr_obj2,ax=ax,ax2=ax2)
        legend_labels=1
        comparison_dir=pathlib.Path(f'{retr_obj.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        output_dir=comparison_dir

    if 'retr_obj3' in kwargs: # compare three retrievals
        suffix='_3'
        retr_obj3=kwargs.get('retr_obj3')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retr_obj3,ax=ax,ax2=ax2)

    leg=ax.legend(fontsize=fs*0.8,ncol=int(math.ceil(len(molecules)/2)),loc='lower left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    for line in leg.get_lines():
        line.set_linestyle('-')
    ax.add_artist(leg)
    if comp_equ==True or 'retr_obj2' in kwargs:
        leg2=ax.legend(handles=chemleg,fontsize=fs*0.8,loc='upper left')
        ax.add_artist(leg2)
    
    ax2.axis('off')
    ax2.set_facecolor('none')
    ax.set(xlabel='VMR', ylabel='Pressure [bar]',yscale='log',xscale='log',
        ylim=(np.max(pressure),np.min(pressure)),xlim=(xmin,xmax))   
    ax.tick_params(labelsize=fs)
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax.set_xlabel('VMR', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{prefix}VMRs_selected{suffix}.pdf')
    plt.close()

def CCF_plot_all(retr_obj,molecules,RVs,noiserange=100,vertical_layout=True,**kwargs):

    # plot all CCFs in one big figure
    
    figname='CCFs_all' if 'retr_obj2' not in kwargs else 'comparison/CCFs_both'

    if vertical_layout==True:
        number=len(molecules)
        nrows=number//2+number%2
        fig,axes = plt.subplots(nrows,2,figsize=(5,nrows*1.3),dpi=200,sharex=True)
        for j,molecule in enumerate(molecules):
            ax=axes[j//2,j%2]
            CCF_norm,_,SNR = retr_obj.ccf_acf_dict[molecule]
            if j%2==1:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

            ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
            ax.set_xlim(-300,300)
            ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
            ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
            ax.plot(RVs,CCF_norm,color=retr_obj.color1,label='CCF')
            if 'retr_obj2' in kwargs: 
                retr_obj2=kwargs.get('retr_obj2')
                CCF_norm2,_,SNR2 = retr_obj2.ccf_acf_dict[molecule]
                ax.plot(RVs,CCF_norm2,color=retr_obj2.color1,label='CCF')
            if retr_obj.chemistry=='freechem':
                molecule_name=retr_obj.parameters.param_mathtext[f'log_{molecule}'][4:] # remove log_
            elif retr_obj.chemistry in ['equchem','quequchem']:
                if molecule=='13CO':
                    molecule_name=r'$^{13}$CO'
                elif molecule=='H2(18)O':
                    molecule_name=r'log H$_2^{18}$O'
            if 'retr_obj2' in kwargs:
                molecule_label=f'{molecule}'
            else:
                molecule_label=f'{molecule}\nS/N={np.round(SNR,decimals=1)}'
            ax.text(0.05, 0.9, molecule_label,transform=ax.transAxes,fontsize=10,verticalalignment='top')

        # in case of odd number, remove last plot
        if len(molecules)%2==1:
            axes[-1,-1].axis('off')

        if 'retr_obj2' in kwargs:
            lines = [Line2D([0], [0], color=retr_obj.color1, linewidth=2,label=retr_obj.target.name),
                Line2D([0], [0], color=retr_obj2.color1, linewidth=2,label=retr_obj2.target.name)]
            leg=axes[0,0].legend(handles=lines,fontsize=11,ncol=2,bbox_to_anchor=(1,1.4),loc='upper center')
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_alpha(None)
            leg.get_frame().set_facecolor((0, 0, 0, 0))
            leg.get_frame().set_edgecolor((0, 0, 0, 0))

        plt.subplots_adjust(wspace=0, hspace=0)


    else:
        number=len(molecules)
        ncols=number//2+number%2
        fig,axes = plt.subplots(2,ncols,figsize=(ncols*2,4),dpi=200,sharex=True,constrained_layout=True)

        for j,molecule in enumerate(molecules):
            ax = axes[j%2,j//2]
            CCF_norm,_,SNR = retr_obj.ccf_acf_dict[molecule]

            ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
            #ax.set_xlim(np.min(RVs),np.max(RVs))
            ax.set_xlim(-300,300)
            ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
            ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
            ax.plot(RVs,CCF_norm,color=retr_obj.color1,label='CCF')
            if 'retr_obj2' in kwargs: 
                retr_obj2=kwargs.get('retr_obj2')
                CCF_norm2,_,SNR2 = retr_obj2.ccf_acf_dict[molecule]
                ax.plot(RVs,CCF_norm2,color=retr_obj2.color1,label='CCF')
            if retr_obj.chemistry=='freechem':
                molecule_name=retr_obj.parameters.param_mathtext[f'log_{molecule}'][4:] # remove log_
            elif retr_obj.chemistry in ['equchem','quequchem']:
                if molecule=='13CO':
                    molecule_name=r'$^{13}$CO'
                elif molecule=='H2(18)O':
                    molecule_name=r'log H$_2^{18}$O'
            if 'retr_obj2' in kwargs:
                molecule_label=f'{molecule_name}'
            else:
                molecule_label=f'{molecule_name}\nS/N={np.round(SNR,decimals=1)}'
            ax.text(0.05, 0.9, molecule_label,transform=ax.transAxes,fontsize=10,verticalalignment='top')

        # in case of odd number, remove last plot
        if len(molecules)%2==1:
            axes[-1,-1].axis('off')

        if 'retr_obj2' in kwargs:
            lines = [Line2D([0], [0], color=retr_obj.color1, linewidth=2,label=retr_obj.target.name),
                Line2D([0], [0], color=retr_obj2.color1, linewidth=2,label=retr_obj2.target.name)]
            leg=axes[0,(int(len(molecules)/2)//2)].legend(handles=lines,fontsize=12,ncol=2,bbox_to_anchor=(0.47,1.4),loc='upper center')
            leg.get_frame().set_linewidth(0.0)
        fig.tight_layout()

    fig.supxlabel(r'$v_{\rm rad}$ [km/s]')
    fig.supylabel('S/N')
    fig.savefig(f'{retr_obj.output_dir}/{figname}.pdf', bbox_inches='tight')
    plt.close()