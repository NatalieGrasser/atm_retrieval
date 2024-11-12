import getpass
import os
if getpass.getuser() == "grasser": # when runnig from LEM
    import atm_retrieval.cloud_cond as cloud_cond
    from atm_retrieval.pRT_model import pRT_spectrum

elif getpass.getuser() == "natalie": # when testing from my laptop
    import cloud_cond as cloud_cond
    from pRT_model import pRT_spectrum
  
import numpy as np
import corner
import matplotlib.pyplot as plt
from labellines import labelLines
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import warnings
import pathlib
import pandas as pd
from petitRADTRANS import Radtrans
warnings.filterwarnings("ignore", category=UserWarning) 

def scale_between(ymin,ymax,arr):
    scale=(ymax-ymin)/(np.nanmax(arr)-np.nanmin(arr))
    scaled_arr=scale*(arr-np.nanmin(arr))+ymin
    return scaled_arr

def plot_spectrum_inset(retrieval_object,inset=True,fs=10,**kwargs):

    wave=retrieval_object.data_wave
    flux=retrieval_object.data_flux
    err=retrieval_object.data_err
    flux_m=retrieval_object.model_flux

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(9.5,3),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for order in range(7):
        # add error for scale
        errmean=np.nanmean(err[order]*retrieval_object.params_dict['s2_ij'][order].reshape(3,1))
        if np.nansum(flux[order])!=0: # skip empty orders
            #ax[1].fill_between([np.min(wave[order]),np.max(wave[order])],-errmean,errmean,color='k',alpha=0.15)
            ax[1].errorbar(np.min(wave[order])-5, 0, yerr=errmean, ecolor=retrieval_object.color1, 
                           elinewidth=1, capsize=2)

        for det in range(3):
            lower=flux[order,det]-err[order,det]*retrieval_object.params_dict['s2_ij'][order,det]
            upper=flux[order,det]+err[order,det]*retrieval_object.params_dict['s2_ij'][order,det]
            ax[0].plot(wave[order,det],flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            ax[0].fill_between(wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax[0].plot(wave[order,det],flux_m[order,det],lw=0.8,alpha=0.8,c=retrieval_object.color1,label='model')
            
            ax[1].plot(wave[order,det],flux[order,det]-flux_m[order,det],lw=0.8,c=retrieval_object.color1,label='residuals')
            if order==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval_object.color1, linewidth=2,label='Bestfit')]
                        #Line2D([0], [0], color=retrieval_object.color2, linewidth=2,label='Residuals')]
                ax[0].legend(handles=lines,fontsize=fs) # to only have it once
        #ax[1].plot(wave[order].flatten(),np.zeros_like(wave[order].flatten()),lw=0.8,alpha=0.5,c='k')
        ax[1].plot([np.min(wave[order]),np.max(wave[order])],[0,0],lw=0.8,alpha=1,c='k')
        
    ax[0].set_ylabel('Normalized Flux',fontsize=fs)
    ax[1].set_ylabel('Residuals',fontsize=fs)
    ax[0].set_xlim(np.min(wave)-10,np.max(wave)+10)
    ax[1].set_xlim(np.min(wave)-10,np.max(wave)+10)
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)

    if inset==True:
        ord=5 
        axins = ax[0].inset_axes([0,-1.3,1,0.8])
        for det in range(3):
            lower=flux[ord,det]-err[ord,det]*retrieval_object.params_dict['s2_ij'][ord,det]
            upper=flux[ord,det]+err[ord,det]*retrieval_object.params_dict['s2_ij'][ord,det]
            axins.fill_between(wave[ord,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            axins.plot(wave[ord,det],flux[ord,det],lw=0.8,c='k')
            axins.plot(wave[ord,det],flux_m[ord,det],lw=0.8,c=retrieval_object.color1,alpha=0.8)
        x1, x2 = np.min(wave[ord]),np.max(wave[ord])
        axins.set_xlim(x1, x2)
        box,lines=ax[0].indicate_inset_zoom(axins,edgecolor="black",alpha=0.2,lw=0.8,zorder=1e3)
        axins.set_ylabel('Normalized Flux',fontsize=fs)
        axins.tick_params(labelsize=fs)
        ax[1].set_facecolor('none') # to avoid hiding lines
        ax[0].set_xticks([])
        
        axins2 = axins.inset_axes([0,-0.3,1,0.3])
        for det in range(3):
            axins2.plot(wave[ord,det],flux[ord,det]-flux_m[ord,det],lw=0.8,c=retrieval_object.color1)
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
        name = 'bestfit_spectrum_inset' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}bestfit_spectrum_inset'
        fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',
                    bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retrieval_object,overplot_species=None):

    if overplot_species!=None: # overplot species onto residuals
        opacities={}
        for spec in overplot_species:
            opa_orders=[]
            wave_orders=[]
            for order in range(7):
                wl_pad=0#7 # wavelength padding because spectrum is not wavelength shifted yet
                wlmin=np.min(retrieval_object.K2166[order])-wl_pad
                wlmax=np.max(retrieval_object.K2166[order])+wl_pad
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
            opacities[spec]=opa_orders

    retrieval=retrieval_object
    residuals=(retrieval.data_flux-retrieval.model_flux)
    fig,ax=plt.subplots(20,1,figsize=(10,13),dpi=200,gridspec_kw={'height_ratios':[2,0.9,0.57]*6+[2,0.9]})
    x=0
    for order in range(7): 
        ax1=ax[x]
        ax2=ax[x+1]
        
        if x!=18: # last ax cannot be spacer, or xlabel also invisible
            ax3=ax[x+2] #for spacing
        for det in range(3):
            ax1.plot(retrieval.data_wave[order,det],retrieval.data_flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            lower=retrieval.data_flux[order,det]-retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det]
            upper=retrieval.data_flux[order,det]+retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det]
            ax1.fill_between(retrieval.data_wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax1.plot(retrieval.data_wave[order,det],retrieval.model_flux[order,det],lw=0.8,alpha=0.8,c=retrieval_object.color1,label='model')
            ax1.set_xlim(np.nanmin(retrieval.data_wave[order])-1,np.nanmax(retrieval.data_wave[order])+1)
            
            ax2.plot(retrieval.data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c=retrieval_object.color1,label='residuals')
            ax2.set_xlim(np.nanmin(retrieval.data_wave[order])-1,np.nanmax(retrieval.data_wave[order])+1)

            # add error for scale
            errmean=np.nanmean(retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det])
            if np.nansum(retrieval.data_flux[order])!=0: # skip empty orders
                ax2.errorbar(np.min(retrieval.data_wave[order,det])-0.3, 0, yerr=errmean, 
                            ecolor=retrieval_object.color1, elinewidth=1, capsize=2)
            
            if x==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval.color1, linewidth=2,label='Bestfit')]
                ax1.legend(handles=lines,fontsize=12,ncol=3,bbox_to_anchor=(0.47,1.4),loc='upper center')
                #leg.get_frame().set_linewidth(0.0)

            ax2.plot([np.min(retrieval.data_wave[order,det]),np.max(retrieval.data_wave[order,det])],[0,0],lw=0.8,c='k')

        if overplot_species!=None:
            colors=['b','r','g']
            wl_shifted=retrieval.model_object.wlshift_orders
            for i,species in enumerate(overplot_species):
                opas=opacities[species]
                ymax=np.nanmax(residuals[order])
                ymin=np.nanmin(residuals[order])
                opa=opas[order]
                opa=scale_between(ymin,ymax,opa)
                ax2.plot(wave_orders[order],opa,lw=0.8,c=colors[i])

        min1=np.nanmin(np.array([retrieval.data_flux[order]-retrieval.data_err[order],retrieval.model_flux[order]]))
        max1=np.nanmax(np.array([retrieval.data_flux[order]+retrieval.data_err[order],retrieval.model_flux[order]]))
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
    name = 'bestfit_spectrum' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}bestfit_spectrum'
    fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf')
    plt.close()

def plot_pt(retrieval_object,fs=12,**kwargs):

    legend_labels=kwargs.get('legend_labels',None)

    if retrieval_object.chemistry in ['equchem','quequchem']:
        C_O = retrieval_object.model_object.params['C/O']
        Fe_H = retrieval_object.model_object.params['Fe/H']
    if retrieval_object.chemistry=='freechem':
        C_O = retrieval_object.model_object.CO
        Fe_H = retrieval_object.model_object.FeH   

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)
    cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    cs_colors=['gold','goldenrod','peru','sandybrown']

    # if pt profile and condensation curve don't intersect, clouds have no effect
    for i,cs in enumerate(cloud_species):
        cs_key = cs[:-3]
        if cs_key == 'KCl':
            cs_key = cs_key.upper()
        P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
        pi=np.where((P_cloud>min(retrieval_object.model_object.pressure))&(P_cloud<max(retrieval_object.model_object.pressure)))[0]
        ax.plot(T_cloud[pi], P_cloud[pi], lw=1.3, label=cloud_labels[i], ls=':',c=cs_colors[i])
    # https://github.com/cphyc/matplotlib-label-lines
    labelLines(ax.get_lines(),align=False,fontsize=fs*0.8,drop_label=True)
    
    # compare with sonora bobcat T=1400K, logg=4.65 -> 10**(4.65)/100 =  446 m/s²
    file=np.loadtxt('t1400g562nc_m0.0.dat')
    pres=file[:,1] # bar
    temp=file[:,2] # K
    ax.plot(temp,pres,linestyle='dashdot',c='blueviolet',linewidth=2)

    if retrieval_object.target.name=='2M0355': # compare with Zhang2022 science verification
        PT_Zhang=np.loadtxt(f'{retrieval_object.target.name}/2M0355_PT_Zhang2021.dat')
        p_zhang=PT_Zhang[:,0]
        t_zhang=PT_Zhang[:,1]
        ax.plot(t_zhang,p_zhang,linestyle='dashdot',c='cornflowerblue',linewidth=2)

    if 'retrieval_object2' in kwargs: # if compare two retrievals, specify object name in legend
        object_label=f'{retrieval_object.target.name} retrieval'
        contr_label=f'{retrieval_object.target.name} contribution'
    else:
        object_label='This retrieval'
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
            lines.append(Line2D([0],[0],marker='o',color=retrieval_object.color1,markerfacecolor=retrieval_object.color1,
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

    xmin,xmax=plot_temperature(retrieval_object,ax,object_label)
       
    if retrieval_object.target.name=='2M0355':
        lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    
    if 'retrieval_object2' in kwargs: # compare two retrievals
        retrieval_object2=kwargs.get('retrieval_object2')
        object_label2=f'{retrieval_object2.target.name} retrieval'
        xmin2,xmax2=plot_temperature(retrieval_object2,ax,object_label2)
        #lines = lines[:1]+[lines[-1]]+lines[1:-1] # move object2 to second position in legend instead of last
        xmin=np.nanmin([xmin,xmin2])
        xmax=np.nanmax([xmax,xmax2])
        model_object2=pRT_spectrum(retrieval_object2,contribution=True)
        model_object2.make_spectrum()
        summed_contr2=np.nanmean(model_object2.contr_em_orders,axis=0) # average over all orders
        contribution_plot2=summed_contr2/np.max(summed_contr2)*(xmax-xmin)+xmin
        ax.plot(contribution_plot2,retrieval_object2.model_object.pressure,linestyle='dashed',
                lw=1.5,alpha=0.8,color=retrieval_object2.color2)
        lines.append(Line2D([0], [0], color=retrieval_object2.color2, alpha=0.8,linewidth=1.5, 
                            linestyle='--',label=f'{retrieval_object2.target.name} contribution'))
        #if retrieval_object2.target.name=='2M0355':
            #lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    
    if 'retrieval_object3' in kwargs:
        retrieval_object3=kwargs.get('retrieval_object3')
        object_label3=f'{retrieval_object3.target.name} retrieval'
        xmin3,xmax3=plot_temperature(retrieval_object3,ax,object_label3)
        xmin=np.nanmin([xmin,xmin2,xmin3])
        xmax=np.nanmax([xmax,xmax2,xmax3])
        model_object3=pRT_spectrum(retrieval_object3,contribution=True)
        model_object3.make_spectrum()
        summed_contr3=np.nanmean(model_object3.contr_em_orders,axis=0) # average over all orders
        contribution_plot3=summed_contr3/np.max(summed_contr3)*(xmax-xmin)+xmin
        ax.plot(contribution_plot3,retrieval_object3.model_object.pressure,linestyle='dashed',
                lw=1.5,alpha=0.8,color=retrieval_object3.color2)
        lines.append(Line2D([0], [0], color=retrieval_object3.color2, alpha=0.8,linewidth=1.5, 
                            linestyle='--',label=f'{retrieval_object3.target.name} contribution'))

    model_object=pRT_spectrum(retrieval_object,contribution=True)
    model_object.make_spectrum()
    summed_contr=np.nanmean(model_object.contr_em_orders,axis=0) # average over all orders
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.model_object.pressure,linestyle='dashed',
            lw=1.5,alpha=0.8,color=retrieval_object.color2)
    lines.append(Line2D([0], [0], color=retrieval_object.color2, alpha=0.8,
                        linewidth=1.5, linestyle='--',label=contr_label))
    lines = lines[:1]+[lines[-1]]+lines[1:-1] # move to second position in legend instead of last
    
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retrieval_object.model_object.pressure),
        np.nanmin(retrieval_object.model_object.pressure)),xlim=(xmin,xmax))
    
    if legend_labels!=None:
        lines=[]
        retr_objects=[retrieval_object,retrieval_object2,retrieval_object3]
        for r,l in zip(retr_objects,legend_labels):
            lines.append(Line2D([0], [0], color=r.color1,linewidth=2,linestyle='-',label=l))
            #lines.append(Line2D([0], [0], color=r.color2, alpha=0.8,linewidth=1.5, 
                            #linestyle='--',label=f'{l} cont.'))
            
    lines.append(Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=1400\,$K, log$\,g=4.75$'))
    
    ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        name = 'PT_profile' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}PT_profile'
        fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf')
        plt.close()

def cornerplot(retrieval_object,getfig=False,figsize=(20,20),fs=12,plot_label='',
            only_abundances=False,only_params=None,not_abundances=False):
    
    plot_posterior=retrieval_object.posterior # posterior that we plot here, might get clipped
    medians,_,_=retrieval_object.get_quantiles(retrieval_object.posterior)
    labels=list(retrieval_object.parameters.param_mathtext.values())
    indices=np.linspace(0,len(retrieval_object.parameters.params)-1,len(retrieval_object.parameters.params),dtype=int)
    
    if only_abundances==True: # plot only abundances
        plot_label='_abundances'
        indices=[]
        for key in retrieval_object.chem_species:
            idx=list(retrieval_object.parameters.params).index(key)
            indices.append(idx)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in indices]).T
        labels=np.array([labels[i] for i in indices])
        medians=np.array([medians[i] for i in indices])

    if only_params is not None: # keys of specified parameters to plot
        indices=[]
        for key in only_params:
            idx=list(retrieval_object.parameters.params).index(key)
            indices.append(idx)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in indices]).T
        labels=np.array([labels[i] for i in indices])
        medians=np.array([medians[i] for i in indices])

    if not_abundances==True: # plot all except abundances
        plot_label='_rest'
        abund_indices=[]
        for key in retrieval_object.chem_species:
            idx=list(retrieval_object.parameters.params).index(key)
            abund_indices.append(idx)
        set_diff = np.setdiff1d(indices,abund_indices)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in set_diff]).T
        labels=np.array([labels[i] for i in set_diff])
        medians=np.array([medians[i] for i in set_diff])
        indices=set_diff

    fig = plt.figure(figsize=figsize) # fix size to avoid memory issues
    fig = corner.corner(plot_posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retrieval_object.color1,
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

    #corner.overplot_lines(fig,medians,color=retrieval_object.color2,lw=1.3,linestyle='solid') # plot median values of posterior

    #if retrieval_object.bestfit_params is not None:
        #corner.overplot_lines(fig,np.array([retrieval_object.bestfit_params[i] for i in indices]),color='b',lw=1.3,linestyle='solid')

    # add true values of test spectrum, plotting didn't work bc x-axis range so small, some didn't show up
    if retrieval_object.target.name in ['test','test_corr']:
        from testspec import test_parameters,test_mathtext
        compare=np.full(len(labels),None) # =None for non-input values of test spectrum
        for key_i in test_parameters.keys():
            label_i=test_mathtext[key_i]
            value_i=test_parameters[key_i]
            if label_i in labels:
                #print(label_i)
                #print(np.where(labels==label_i))
                i=np.where(labels==label_i)[0][0]
                compare[i]=value_i # add only those values that are used in cornerplot, in correct order
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'{compare[i]}'
            fig.axes[x].title.set_text(titles[x])
            x+=len(labels)+1

    plt.subplots_adjust(wspace=0,hspace=0)

    if getfig==False:
        name= f'cornerplot{plot_label}' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}cornerplot{plot_label}'
        fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
    else:
        ax = np.array(fig.axes)
        return fig, ax

def make_all_plots(retrieval_object,only_abundances=False,only_params=None,split_corner=True):
    plot_spectrum_split(retrieval_object)
    plot_spectrum_inset(retrieval_object)
    plot_pt(retrieval_object)
    summary_plot(retrieval_object)
    opacity_plot(retrieval_object)
    if retrieval_object.chemistry=='freechem':
        VMR_plot(retrieval_object,comp_equ=True) # compare with what equchem abundances would be like
        ratios_cornerplot(retrieval_object) # already in equchem cornerplot by default
        if split_corner: # split corner plot to avoid massive files
            cornerplot(retrieval_object,only_abundances=True)
            cornerplot(retrieval_object,not_abundances=True)
        else: # make cornerplot with all parameters, could be huge, avoid this
            cornerplot(retrieval_object,only_params=only_params)
    elif retrieval_object.chemistry in ['equchem','quequchem']:
        VMR_plot(retrieval_object)
        if split_corner: # split corner plot to avoid massive files
            only_params=['rv','vsini','log_g','C/O','Fe/H',
                         'log_C12_13_ratio','log_O16_18_ratio','log_O16_17_ratio']
            if retrieval_object.chemistry=='quequchem':
                for val in ['log_Pqu_CO_CH4','log_Pqu_NH3','log_Pqu_HCN']:
                    only_params.append(val)
            cornerplot(retrieval_object,only_params=only_params,plot_label='1')
            only_params2=list(set(retrieval_object.parameters.param_keys)-set(only_params))
            cornerplot(retrieval_object,only_params=only_params2,plot_label='2')
        else: # avoid this though
            cornerplot(retrieval_object,only_params=only_params)
    
def summary_plot(retrieval_object):

    fs=13
    if retrieval_object.chemistry in ['equchem','quequchem']:
        only_params=['rv','vsini','log_g','T0','C/O','Fe/H',
                 'log_C12_13_ratio','log_O16_18_ratio','log_O16_17_ratio']
    if retrieval_object.chemistry=='freechem':
        only_params=['rv','vsini','log_g','T0','log_H2O','log_12CO',
                 'log_13CO','log_HF','log_H2(18)O','log_H2S']
    fig, ax = cornerplot(retrieval_object,getfig=True,only_params=only_params,figsize=(17,17),fs=fs)
    l, b, w, h = [0.37,0.84,0.6,0.15] # left, bottom, width, height
    ax_spec = fig.add_axes([l,b,w,h])
    ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
    plot_spectrum_inset(retrieval_object,ax=(ax_spec,ax_res),inset=False,fs=fs)

    l, b, w, h = [0.68,0.47,0.29,0.29] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retrieval_object,ax=ax_PT,fs=fs)
    name = 'summary' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}summary'
    fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()


def opacity_plot(retrieval_object,only_params=None):
    Kband=retrieval_object.target.K2166

    # plot 6 most abundant species
    only_params=[]
    abunds=[]
    pRT_names=[]
    labels=[]
    species=retrieval_object.chem_species

    if retrieval_object.chemistry=='freechem':
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        for spec in species:
            abunds.append(retrieval_object.params_dict[spec])
        
    elif retrieval_object.chemistry in ['equchem','quequchem']:
        species_info = pd.read_csv(os.path.join('species_info.csv'))
        for spec in species:
            model_object=pRT_spectrum(retrieval_object)    
            mass_fractions=model_object.mass_fractions
            MMW=model_object.MMW
            for spec in retrieval_object.species:
                mass=species_info.loc[species_info["pRT_name"]==spec]['mass'].values[0]
                abunds.append(np.median(mass_fractions[spec]*MMW/mass)) # take median of abundance
                
    abunds, species = zip(*sorted(zip(abunds, species)))
    only_params=species[-6:][::-1] # get largest 6
    abunds = abunds[-6:][::-1] # get largest 6
    VMRs=[]
    colors=[]
    if retrieval_object.chemistry=='freechem':
        for i,par in enumerate(only_params):
            pRT_names.append(species_info.loc[par[4:],'pRT_name'])
            labels.append(species_info.loc[par[4:],'mathtext_name'])
            VMRs.append(10**retrieval_object.params_dict[only_params[i]])
            colors.append(species_info.loc[f'{only_params[i][4:]}','color'])
    elif retrieval_object.chemistry in ['equchem','quequchem']:
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
    
    T = np.array([1400]).reshape(1)
    wave_cm, opas = atmosphere.get_opa(T)
    wave_nm = wave_cm*1e7
    ymin,ymax=5e-8,5e2

    fig,ax=plt.subplots(1,1,figsize=(6,3),dpi=200)
    lines=[]
    for i,m in enumerate(pRT_names):
        #abund=10**retrieval_object.params_dict[only_params[i]]
        #col=species_info.loc[f'{only_params[i][4:]}','color']
        #print(names[i],abund)
        spec,=plt.plot(wave_nm,opas[m]*VMRs[i],lw=0.5,c=colors[i])
        lines.append(Line2D([0],[0],color=spec.get_color(),
                        linewidth=2,label=labels[i]))
        
    for order in range(7):
        for det in range(3):
            plt.fill_betweenx([ymin,ymax],Kband[order,det][0],Kband[order,det][1],color='k',alpha=0.07)
    plt.yscale('log')
    plt.ylabel('Opacity [cm$^2$/g]')
    plt.xlabel("Wavelength [nm]")
    plt.xlim(np.min(retrieval_object.target.K2166),np.max(retrieval_object.target.K2166))
    plt.ylim(ymin,ymax)
    legend=plt.legend(handles=lines,ncol=3,loc='upper center')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((0, 0, 0, 0))
    name = 'opacities' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}opacities'
    fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()
    del opas # to avoid memory issues

def CCF_plot(retrieval_object,molecule,RVs,CCF_norm,ACF_norm,noiserange=50):
    SNR=CCF_norm[np.where(RVs==0)[0][0]]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5,3.5),dpi=200,gridspec_kw={'height_ratios':[3,1]})
    for ax in (ax1,ax2):
        ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
        ax.set_xlim(np.min(RVs),np.max(RVs))
        ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
        ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
    ax1.plot(RVs,CCF_norm,color=retrieval_object.color1,label='CCF')
    ax1.plot(RVs,ACF_norm,color=retrieval_object.color1,linestyle='dashed',alpha=0.5,label='ACF')
    ax1.set_ylabel('S/N')
    ax1.legend(loc='upper right')
    if retrieval_object.chemistry=='freechem':
        molecule_name=retrieval_object.parameters.param_mathtext[f'log_{molecule}'][4:] # remove log_
    elif retrieval_object.chemistry in ['equchem','quequchem']:
        if molecule=='13CO':
            molecule_name=r'$^{13}$CO'
        elif molecule=='H2(18)O':
            molecule_name=r'log H$_2^{18}$O'
    molecule_label=f'{molecule_name}\nS/N={np.round(SNR,decimals=1)}'
    #molecule_label=f'{molecule_name}'
    ax1.text(0.05, 0.9, molecule_label,transform=ax1.transAxes,fontsize=14,verticalalignment='top')
    ax2.plot(RVs,CCF_norm-ACF_norm,color=retrieval_object.color1)
    ax2.set_ylabel('CCF-ACF')
    ax2.set_xlabel(r'$v_{\rm rad}$ (km/s)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f'{retrieval_object.output_dir}/CCF_{molecule}.pdf')
    plt.close()

def compare_retrievals(retrieval_object1,retrieval_object2,fs=12,**kwargs): # compare cornerplot+PT of two retrievals

    legend_labels=kwargs.get('legend_labels',None)
    num=2 # number of retrievals
    # can only compare freechem+freechem or freechem+equchem/quequchem(+equchem/quequchem)
    if retrieval_object1.chemistry=='freechem' and retrieval_object2.chemistry=='freechem':
        only_params=['log_H2O','log_12CO','log_13CO','log_CH4',
                    'log_NH3','log_HCN','log_HF','log_H2(18)O','log_H2S']
        
        labels=list(retrieval_object1.parameters.param_mathtext.values())
        indices=[]
        for key in only_params: # keys of specified parameters to plot
            idx=list(retrieval_object1.parameters.params).index(key)
            indices.append(idx)
        posterior1=np.array([retrieval_object1.posterior[:,i] for i in indices]).T
        posterior2=np.array([retrieval_object2.posterior[:,i] for i in indices]).T
        labels=np.array([labels[i] for i in indices])

    elif retrieval_object1.chemistry=='freechem' and retrieval_object2.chemistry in ['equchem','quequchem']:

        only_params=['C/O','Fe/H','log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio']
        # compare ratio posterios, must be in same order!!
        posterior1=retrieval_object1.ratios_posterior
        # add log_O16_18_ratio (last one) to equchem again, bc freechem has C18O and H218O ratios
        reshaped=retrieval_object2.ratios_posterior[:,-1].reshape(len(retrieval_object2.ratios_posterior[:,-1]),1)
        posterior2=np.hstack([retrieval_object2.ratios_posterior,reshaped])
        labels=['C/O','[Fe/H]',r'log $^{12}$CO/$^{13}$CO',r'log $^{12}$CO/C$^{17}$O',
                r'log $^{12}$CO/C$^{18}$O',r'log H$_2^{16}$O/H$_2^{18}$O']

        if 'retrieval_object3' in kwargs:
            num=3
            retrieval_object3=kwargs.get('retrieval_object3')
            reshaped=retrieval_object3.ratios_posterior[:,-1].reshape(len(retrieval_object3.ratios_posterior[:,-1]),1)
            posterior3=np.hstack([retrieval_object3.ratios_posterior,reshaped])

    #figsize=15
    figsize=14
    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues

    def plot_corner(posterior,retr_obj,labels,fig,getfig=False):
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
                        quiet=True)
        
        titles = [axi.title.get_text() for axi in fig.axes]
        if getfig:
            return fig,titles
        else:
            return titles
        
    fig,titles1=plot_corner(posterior1,retrieval_object1,labels,fig,getfig=True)
    titles2=plot_corner(posterior2,retrieval_object2,labels,fig)
    enum=[0,1]
    titles_list=[titles1,titles2]
    colors_list=[retrieval_object1.color1,retrieval_object2.color1]

    # overplot medians as solid lines
    #corner.overplot_lines(fig,np.array([retrieval_object1.bestfit_params[i] for i in indices]),
                          #color='b',lw=1.3,linestyle='solid')


    if 'retrieval_object3' in kwargs:
        titles3=plot_corner(posterior3,retrieval_object3,labels,fig)
        enum=[0,1,2]
        titles_list.append(titles3)
        colors_list.append(retrieval_object3.color1)

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
                #y=1.6-(0.2*(run+1))
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

    l, b, w, h = [0.58,0.65,0.39,0.39] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])

    if 'retrieval_object3' not in kwargs:
        plot_pt(retrieval_object1,retrieval_object2=retrieval_object2,ax=ax_PT)
    else:
        plot_pt(retrieval_object1,retrieval_object2=retrieval_object2,
                ax=ax_PT,retrieval_object3=retrieval_object3,legend_labels=legend_labels)

    comparison_dir=pathlib.Path(f'{retrieval_object1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(f'{comparison_dir}/cornerplot_{num}.pdf',bbox_inches="tight",dpi=200)
    plt.close()

def compare_two_CCFs(retrieval_object1,retrieval_object2,molecules,noiserange=50):

    RVs=np.arange(-500,500,1) # km/s
    comparison_dir=pathlib.Path(f'{retrieval_object1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for i,molecule in enumerate(molecules):

        CCF_norm1=retrieval_object1.CCF_list[i]
        ACF_norm1=retrieval_object1.ACF_list[i]
        SNR1=CCF_norm1[np.where(RVs==0)[0][0]]
        CCF_norm2=retrieval_object2.CCF_list[i]
        ACF_norm2=retrieval_object2.ACF_list[i]
        SNR2=CCF_norm2[np.where(RVs==0)[0][0]]

        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5,3.5),dpi=200,gridspec_kw={'height_ratios':[3,1]})
        for ax in (ax1,ax2):
            ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
            ax.set_xlim(np.min(RVs),np.max(RVs))
            ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
            ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
        ax1.plot(RVs,CCF_norm1,color=retrieval_object1.color1,label=f'{retrieval_object1.target.name}')
        ax1.plot(RVs,ACF_norm1,color=retrieval_object1.color1,linestyle='dashed',alpha=0.5)

        ax1.plot(RVs,CCF_norm2,color=retrieval_object2.color1,label=f'{retrieval_object2.target.name}')
        ax1.plot(RVs,ACF_norm2,color=retrieval_object2.color1,linestyle='dashed',alpha=0.5)

        lines = [Line2D([0], [0], color=retrieval_object1.color1,linewidth=2,label=f'{retrieval_object1.target.name}'),
                 Line2D([0], [0], color=retrieval_object2.color1,linewidth=2,label=f'{retrieval_object2.target.name}'),
                 Line2D([0], [0], color='k',linewidth=2,alpha=0.5,label='CCF'),
                 Line2D([0], [0], color='k',linestyle='--',linewidth=2,alpha=0.2,label='ACF')]
        ax1.legend(handles=lines,fontsize=9,loc='upper right')
        ax1.set_ylabel('S/N')

        molecule_label=str(retrieval_object1.parameters.param_mathtext[f'log_{molecule}'][4:]) # remove log_
        ax1.text(0.05, 0.9, molecule_label,transform=ax1.transAxes,fontsize=14,verticalalignment='top')

        ax2.plot(RVs,CCF_norm1-ACF_norm1,color=retrieval_object1.color1)
        ax2.plot(RVs,CCF_norm2-ACF_norm2,color=retrieval_object2.color1)
        ax2.set_ylabel('CCF-ACF')
        ax2.set_xlabel(r'$v_{\rm rad}$ (km/s)')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        filename=f'CCF_{molecule}_2.pdf'
        fig.savefig(f'{comparison_dir}/{filename}',bbox_inches="tight",dpi=200)
        plt.close()

def ratios_cornerplot(retrieval_object,fs=10,**kwargs):

    if retrieval_object.chemistry in ['equchem','quequchem']:
        labels=['C/O','[Fe/H]',r'log $^{12}$C/$^{13}$C',r'log $^{16}$O/$^{17}$O',r'log $^{16}$O/$^{18}$O']
    elif retrieval_object.chemistry=='freechem':
        labels=['C/O','[C/H]',r'log $^{12}$CO/$^{13}$CO',r'log $^{12}$CO/C$^{17}$O',
                r'log $^{12}$CO/C$^{18}$O',r'log H$_2^{16}$O/H$_2^{18}$O']
    
    fig = plt.figure(figsize=(8,8)) # fix size to avoid memory issues
    fig = corner.corner(retrieval_object.ratios_posterior,
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retrieval_object.color1,
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
    
    if 'retrieval_object2' in kwargs: # compare two retrievals
        retrieval_object2=kwargs.get('retrieval_object2')
        corner.corner(retrieval_object2.ratios_posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retrieval_object2.color1,
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

        for run,titles_list,color in zip([0,1],[titles,titles2],[retrieval_object.color1,retrieval_object2.color1]):
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
        comparison_dir=pathlib.Path(f'{retrieval_object.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        filename=f'{comparison_dir}/ratios_2.pdf'

    else:
        for i, title in enumerate(titles):
            if len(title) > 1: # change 30 to 1 if you want all titles to be split
                title_split = title.split('=')
                titles[i] = title_split[0] + '\n ' + title_split[1]
            fig.axes[i].title.set_text(titles[i])
        name1=f'{retrieval_object.output_dir}/ratios.pdf'
        name2=f'{retrieval_object.output_dir}/{retrieval_object.callback_label}ratios.pdf'
        filename = name1 if retrieval_object.callback_label=='final_' else name2

    for i, axi in enumerate(fig.axes):
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)

    plt.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(filename,bbox_inches="tight",dpi=200)
    plt.close()

def VMR_plot(retrieval_object,molecules='all',fs=10,comp_equ=False,**kwargs):

    #prefix=retrieval_object.callback_label if retrieval_object.callback_label=='final_' else ''
    prefix=''
    suffix=''
    output_dir=retrieval_object.output_dir
    fig,ax=plt.subplots(1,1,figsize=(6,4),dpi=200)
    species_info = pd.read_csv(os.path.join('species_info.csv'))
    molecules=molecules if molecules!='all' else ['H2','He','H2O','H2(18)O','12CO','13CO','CH4','H2S','HCN','NH3']
    alpha=0.6 if 'retrieval_object2' in kwargs or comp_equ==True else 1
    legend_labels=0
    xmin,xmax=1e-11,1.3

    def plot_VMRs(retr_obj,ax,ax2):
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
                linestyle='dashed'
                ax.plot(VMR,pressure,label=label,linestyle='dashed',c=color)
            elif retr_obj.chemistry in ['equchem','quequchem']:
                if retr_obj.chemistry=='equchem':
                    linestyle='solid'
                if retr_obj.chemistry=='quequchem':
                    linestyle='dotted'
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

        model_object=pRT_spectrum(retr_obj,contribution=True)
        model_object.make_spectrum()
        summed_contr=np.nanmean(model_object.contr_em_orders,axis=0) # average over all orders
        contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
        ax2.plot(contribution_plot,np.log10(retr_obj.model_object.pressure)[::-1],
                    lw=1,alpha=alpha*0.5,color=retr_obj.color1,linestyle=linestyle)
        ax2.set_xlim(np.min(contribution_plot),np.max(contribution_plot))

    pressure=retrieval_object.model_object.pressure
    ax2 = ax.inset_axes([0,0,1,1]) # [x0, y0, width, height] , for emission contribution

    plot_VMRs(retrieval_object,ax=ax,ax2=ax2)
    legend_labels=1 # only make legend labels once 

    # compare freechem VMRs to equilibrium chemistry with other retrieved params remainig equal
    if comp_equ==True:
        if getpass.getuser() == "grasser": # when runnig from LEM
            from atm_retrieval.retrieval import Retrieval
            from atm_retrieval.parameters import Parameters
        elif getpass.getuser() == "natalie": # when testing from my laptop
            from retrieval import Retrieval
            from parameters import Parameters

        parameters_equ = retrieval_object.params_dict
        parameters_equ.update({'C/O': retrieval_object.params_dict['C/O'],
                        'Fe/H': retrieval_object.params_dict['C/H'],
                        'log_C12_13_ratio': retrieval_object.params_dict['log_12CO/13CO'],
                        'log_O16_18_ratio': retrieval_object.params_dict['log_H2O/H2(18)O'],
                        'log_O16_17_ratio': retrieval_object.params_dict['log_12CO/C17O']})
        parameters_equ = Parameters({}, parameters_equ)
        parameters_equ.param_priors['log_l']=[-3,0]
        retrieval_equ = Retrieval(target=retrieval_object.target,parameters=parameters_equ, 
                                  output_name=retrieval_object.output_name,
                                chemistry='equchem',PT_type=retrieval_object.PT_type)
        retrieval_equ.model_object=pRT_spectrum(retrieval_equ)
        plot_VMRs(retrieval_equ,ax=ax,ax2=ax2)
        suffix='_eq'

    if 'retrieval_object2' in kwargs: # compare two retrievals
        prefix=''
        suffix='_2'
        retrieval_object2=kwargs.get('retrieval_object2')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retrieval_object2,ax=ax,ax2=ax2)
        comparison_dir=pathlib.Path(f'{retrieval_object.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        output_dir=comparison_dir

    if 'retrieval_object3' in kwargs: # compare three retrievals
        suffix='_3'
        retrieval_object3=kwargs.get('retrieval_object3')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retrieval_object3,ax=ax,ax2=ax2)

    leg=ax.legend(fontsize=fs*0.8)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    for line in leg.get_lines():
        line.set_linestyle('-')
    
    ax2.axis('off')
    ax2.set_facecolor('none')
    ax.set(xlabel='VMR', ylabel='Pressure [bar]',yscale='log',xscale='log',
        ylim=(np.max(pressure),np.min(pressure)),xlim=(xmin,xmax))   
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('VMR', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{prefix}VMRs{suffix}.pdf')
    plt.close()




    