import os
import cloud_cond as cloud_cond
from pRT_model import pRT_spectrum
from utils import *
import numpy as np
import corner
import copy
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
import re
import getpass
import pandas as pd
from petitRADTRANS import Radtrans
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings("ignore", category=UserWarning) 
if getpass.getuser() == "grasser": # when runnig from LEM
    path_tables = '/net/lem/data2/regt/fastchem_tables'
elif getpass.getuser() == "natalie": # when testing from my laptop
    path_tables = '/home/natalie/fastchem_tables'
            
def plot_spectrum_inset(retr_obj,inset=True,fs=10,plot_veiling=False,**kwargs):

    wave=retr_obj.data_wave
    flux=retr_obj.data_flux
    err=retr_obj.data_err
    flux_m=np.copy(retr_obj.model_flux)
    suffix=''
    low=[]
    up=[]

    if retr_obj.instrument=='CRIRES':
        wl_unit = 'nm'
        pm_xlim = 10 # in nm
        if retr_obj.primary_label==False:
            for i in range(flux.shape[0]):
                #if np.isnan(flux[i]).all()==False:
                    #fl_med=np.nanmedian(flux[i])
                    #flux[i]/=fl_max
                    #flux_m[i]/=fl_max
                if np.isnan(flux[i]).all():
                    flux_m[i]*=np.nan
    elif retr_obj.instrument=='LIFE':
        wl_unit = r'\mathrm{\mu}m'
        pm_xlim = 0.0 # in um

    if plot_veiling: # show spectrum without veiling for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        #retr_obj2.parameters.params.pop('log_k_rk')
        retr_obj2.parameters.params['log_k_rk']=-6
        retr_obj2.parameters.params['d_rk']=0
        model_wo_veiling = pRT_spectrum(retr_obj2).make_spectrum()
        noveil_c = 'm'
        suffix='_noveil'

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(10,2.5),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for part in range(retr_obj.n_parts):
        if np.isnan(flux[part]).all():
            #flux_m[part]/=np.nanmedian(flux_m[part])
            flux_m[part]*=np.nan # fixed ylim issue for A
            continue
        # add error for scale
        order = slice(part,part+3)
        if part%3==0 and np.nansum(flux[order])!=0 and retr_obj.instrument=='CRIRES': # skip empty orders 
            errmean=np.nanmean(err[order]*retr_obj.params_dict['s2'][order].reshape(3,1))
            ax[1].errorbar(np.min(wave[order])-5, 0, yerr=errmean, ecolor=retr_obj.color,elinewidth=1, capsize=2)

        if retr_obj.instrument=='CRIRES' or len(flux[part])>100: 
            lw=0.8
            ax[0].plot(wave[part],flux[part],lw=lw,alpha=1,c='k',label='Data')
            ax[1].plot(wave[part],flux[part]-flux_m[part],lw=lw,c=retr_obj.color,label='residuals')
            lower=flux[part]-err[part]*retr_obj.params_dict['s2'][part]
            upper=flux[part]+err[part]*retr_obj.params_dict['s2'][part]
            low.append(np.nanmin(lower))
            up.append(np.nanmax(upper))
            ax[0].fill_between(wave[part],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
                    
        elif retr_obj.instrument=='LIFE': 
            size=2.5
            lw=1.5
            elw=0.9
            ax[0].errorbar(wave[part],flux[part],yerr=err,fmt='o',markersize=size,elinewidth=elw,c='k',label='Data')
            ax[1].scatter(wave[part],flux[part]-flux_m[part],s=size,c=retr_obj.color)

        #if np.isnan(flux[part]).all():# scale will be weird, set manually
            #flux_m[part] = scale_between(ymin,ymax,flux_m[part])
        if plot_veiling:
            ax[0].plot(wave[part],model_wo_veiling[part],lw=lw,alpha=0.8,c=noveil_c,label='r$_k$=0')
            rk = np.round(retr_obj.params_dict('d_rk'),decimals=2)
            model_label = f'r$_k$={rk}'
        else:
            model_label = 'Bestfit'
        ax[0].plot(wave[part],flux_m[part],lw=lw,alpha=0.8,c=retr_obj.color,label=model_label)
        if part==0:
            if retr_obj.instrument=='CRIRES':
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                    #mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                    Line2D([0], [0], color=retr_obj.color, linewidth=2,label='Bestfit')]
                ax[0].legend(handles=lines,fontsize=fs) # to only have it once
            elif retr_obj.instrument=='LIFE':
                ax[0].legend(fontsize=fs)
        ax[1].plot([np.min(wave[part]),np.max(wave[part])],[0,0],lw=0.8,alpha=1,c='k')

    ax[0].set_ylabel('Normalized Flux',fontsize=fs)
    ax[1].set_ylabel('Residuals',fontsize=fs)
    ax[0].set_xlim(np.min(wave)-pm_xlim,np.max(wave)+pm_xlim)
    ax[0].set_ylim(np.nanmin(np.array([flux,flux_m])),np.nanmax(np.array([flux,flux_m])))
    ax[1].set_xlim(np.min(wave)-pm_xlim,np.max(wave)+pm_xlim)
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)
    
    if np.isnan(flux[:3]).all():
        ax[0].set_xlim(np.min(wave[3:])-pm_xlim,np.max(wave)+pm_xlim)
        ax[1].set_xlim(np.min(wave[3:])-pm_xlim,np.max(wave)+pm_xlim)
    
    if retr_obj.primary_label==False: # normalized differently, alig w inset lines
        ax[0].set_ylim(np.min(low),np.max(up))

    if inset==True and retr_obj.instrument=='CRIRES':
        suffix='_inset'
        parts = slice(15,18) # 5th order
        axins = ax[0].inset_axes([0,-1.3,1,0.75]) # left, bottom, width, height
        for p in [15,16,17]:
            lower=flux[p]-err[p]*retr_obj.params_dict['s2'][p]#[:, np.newaxis]
            upper=flux[p]+err[p]*retr_obj.params_dict['s2'][p]#[:, np.newaxis]
            axins.fill_between(wave[p],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            axins.plot(wave[p],flux[p],lw=0.8,c='k')
            if plot_veiling:
                axins.plot(wave[p],model_wo_veiling[p],lw=lw,alpha=0.8,c=noveil_c,label='w/o veiling')
            axins.plot(wave[p],flux_m[p],lw=0.8,c=retr_obj.color,alpha=0.8)
        x1, x2 = np.min(wave[parts]),np.max(wave[parts])
        axins.set_xlim(x1, x2)
        box,lines=ax[0].indicate_inset_zoom(axins,edgecolor="black",alpha=0.2,lw=0.8,zorder=1e3)
        axins.set_ylabel('Normalized Flux',fontsize=fs)
        axins.tick_params(labelsize=fs)
        ax[1].set_facecolor('none') # to avoid hiding lines
        ax[0].set_xticks([])
        
        axins2 = axins.inset_axes([0,-0.3,1,0.3])
        axins2.plot(wave[parts].flatten(),flux[parts].flatten()-flux_m[parts].flatten(),lw=0.8,c=retr_obj.color)
        axins2.plot([np.min(wave[parts]),np.max(wave[parts])],[0,0],lw=0.8,alpha=1,c='k')
        axins2.set_xlim(x1, x2)
        axins2.set_xlabel(f'Wavelength [{wl_unit}]',fontsize=fs)
        axins2.set_ylabel('Res.',fontsize=fs)
        tick_spacing=1
        axins2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        axins2.tick_params(labelsize=fs)
    else:
        ax[1].set_xlabel(f'Wavelength [{wl_unit}]',fontsize=fs) # if no inset

    plt.subplots_adjust(wspace=0, hspace=0)
    if 'ax' not in kwargs:
        name = f'bestfit{suffix}' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit{suffix}'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf', bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retr_obj,show_opacities=None,plot_components=False,
                        plot_cloud=False, plot_veiling=False):

    # function only for CRIRES spectra anyway
    crires_shape = (retr_obj.n_orders,retr_obj.n_dets,retr_obj.n_pixels)

    if show_opacities!=None: # overplot species to check features
        opacities={}
        species_string = ''

        if isinstance(show_opacities, list)==False:
            show_opacities=list(show_opacities)

        for spec in show_opacities: 
            species_string+=str(spec)
            opa_orders=[]
            for order in range(7):
                wlen_range=np.array([np.min(retr_obj.target.K2166[order]),np.max(retr_obj.target.K2166[order])])*1e-3 # nm to microns
                atm = Radtrans(line_species=[retr_obj.species_info.loc[spec,'pRT_name']],
                                    rayleigh_species = [],
                                    continuum_opacities = [],
                                    wlen_bords_micron=wlen_range, 
                                    mode='lbl',
                                    lbl_opacity_sampling=3) # take every nth point (=3 in deRegt+2024)
                
                wave_cm, opas = atm.get_opa(np.array([retr_obj.params_dict['T_maxcont']]).reshape(1))
                opa = opas[retr_obj.species_info.loc[spec,'pRT_name']].flatten()

                from astropy import constants as const
                wl_shifted= wave_cm*1e7*(1.0+(retr_obj.params_dict['rv']-retr_obj.target.vbary)/const.c.to('km/s').value)
                waves_even = np.linspace(np.min( wave_cm*1e7), np.max( wave_cm*1e7), np.array(wave_cm).size) # wavelength array has to be regularly spaced
                opa = np.interp(waves_even, wl_shifted, opa)

                parts = slice(order*3,order*3+3)
                opa_interp = np.interp(retr_obj.data_wave[parts].flatten(), waves_even, opa)
                opa_orders.append(opa_interp)
            opacities[spec] = np.array(opa_orders).reshape(crires_shape)
        retr_obj.opacities=opacities

    retr=retr_obj
    residuals=(retr.data_flux-retr.model_flux).reshape(crires_shape)
    to_reshape= [np.copy(retr_obj.data_flux), retr_obj.data_err, retr_obj.data_wave, np.copy(retr_obj.model_flux)]
    data_flux, data_err, data_wave, model_flux = [var.reshape(crires_shape) for var in to_reshape]
    s2 = retr.params_dict['s2'].reshape((crires_shape[:-1]))

    n_sub = 20
    n_gs = 6
    plot_orders = np.linspace(0,6,7,dtype=int)
    if np.isnan(data_flux[0].flatten()).all(): # often full of tellurics, remove subplot if empty
         n_sub-=3
         n_gs = 5
         plot_orders = np.linspace(1,6,6,dtype=int)

    figsize=(10,13)
    gridspec_kw={'height_ratios':[2,0.9,0.57]*n_gs+[2,0.9]}
    #if 'log_k_rk' in retr.params_dict:
        #rk_func = lambda x: 10**retr.params_dict['log_k_rk']*np.array(x) + retr.params_dict['d_rk']
    if plot_components==True:
        primary_flx= retr.model_object.primary_broadened
        secondary_flx = retr.model_object.secondary_flux
        # was not normalized correctly for some reason? norm all in same way
        #for i in range(retr_obj.n_orders):
            #for j in range(retr_obj.n_dets):
                #if np.isnan(data_flux[i,j]).all()==False:
                    #data_flux[i,j]/=np.nanmedian(data_flux[i,j])
                    
        phi_sec = retr_obj.params_dict['phi_secondary']
        print('Phi star=',1-phi_sec)
        print('Phi BD  =',phi_sec)
        #figsize=(9,15)
        gridspec_kw={'height_ratios':[2,0.4,0.57]*n_gs+[2,0.4]}

    if plot_veiling: # show spectrum without veiling for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        mask = np.isfinite(data_flux)
        retr_obj2.parameters.params['log_k_rk']=-6
        retr_obj2.parameters.params['d_rk']=0
        model_wo_veiling = pRT_spectrum(retr_obj2).make_spectrum().reshape(crires_shape)
        model_wo_veiling[~mask] = np.nan
        noveil_c = 'm'

    if plot_cloud: # show spectrum without cloud for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        mask = np.isfinite(data_flux)
        retr_obj2.cloud_mode=None
        model_wo_cloud = pRT_spectrum(retr_obj2).make_spectrum().reshape(crires_shape)
        model_wo_cloud[~mask] = np.nan
        nocloud_c = 'm'

    fig,ax=plt.subplots(n_sub,1,figsize=figsize,dpi=200,gridspec_kw=gridspec_kw)
    x=0
    
    for idx,order in enumerate(plot_orders):
        ax1=ax[x]
        ax2=ax[x+1]
        min_array, max_array = [],[]
        opa_orders = []
        
        if x!=(n_sub-2): # last ax cannot be spacer, or xlabel also invisible
            ax3=ax[x+2] #for spacing
        for det in range(3):
            mask = np.isnan(data_flux[order,det])
            if np.isnan(data_flux[order,det]).all() and retr.primary_label==False:
                #model_sec_flx=np.copy(model_flux[order,det])
                #model_sec_flx/=np.nanmax(model_sec_flx)#*np.nanmean(phi_comp[order,:,-1])
                model_flux[order,det]*=np.nan # don't show as A+B because it's only model of B
            #elif np.isnan(data_flux[order,det]).all() and retr.primary_label==True:
                #model_flux[order,det]/=np.nanmedian(model_flux[order,det]) # show but normalized (at same height as others)

            ax1.plot(data_wave[order,det],data_flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            if plot_veiling:
                if np.isnan(data_flux[order,det]).all()==False:
                    ax1.plot(data_wave[order,det],model_wo_veiling[order,det],lw=0.8,c=noveil_c,label='r$_k$=0')
                    noveil_label='r$_k$=0'
                    rk = np.round(retr_obj.params_dict['d_rk'],decimals=2)
                    model_label = f'r$_k$={rk}'
            elif plot_cloud:
                if np.isnan(data_flux[order,det]).all()==False:
                    ax1.plot(data_wave[order,det],model_wo_cloud[order,det],lw=0.8,c=nocloud_c,label='no cloud')
                    nocloud_label='no cloud'
                    model_label = 'cloud'
            else:
                model_label = 'Bestfit'
            ax1.plot(data_wave[order,det],model_flux[order,det],lw=0.8,alpha=0.8,c=retr_obj.color,label='model')
            ax1.set_xlim(np.nanmin(data_wave[order])-1,np.nanmax(data_wave[order])+1)
            if plot_components==True:
                prim_c='orange'
                sec_c='dodgerblue'
                prim_flx= primary_flx[order,det]
                sec_flx = secondary_flx[order,det]
                
                sec_flx[mask] = np.nan
                ax1.plot(data_wave[order,det], prim_flx, label='A',lw=0.8, c=prim_c)
                ax1.plot(data_wave[order,det], sec_flx,lw=0.8, label='B', c=sec_c)
                
                if np.isfinite(prim_flx).any() and np.isfinite(sec_flx).any():
                    min_array.append(np.nanmin([prim_flx,sec_flx]))
                    max_array.append(np.nanmax([prim_flx,sec_flx]))
                #if np.isnan(data_flux[order,det]).all(): # then model is only secondary flux
                    #ax1.plot(data_wave[order,det], model_sec_flx*avg_B,lw=0.8, c=sec_c)
                    #min_array.append(np.nanmin(model_sec_flx*avg_B))
                    #max_array.append(np.nanmax(model_sec_flx*avg_B))
            #if 'log_k_rk' in retr.params_dict:
                #ax1.plot(data_wave[order,det], rk_func(data_wave[order,det]),lw=0.8, label='D', c='yellowgreen')

            if show_opacities==None: # else show opacities
                ax2.plot(data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c=retr_obj.color,label='residuals')
                ax2.set_xlim(np.nanmin(data_wave[order])-1,np.nanmax(data_wave[order])+1)

            # add error for scale
            if retr.callback_label=='final_':
                lower=data_flux[order,det]-data_err[order,det]*s2[order,det]
                upper=data_flux[order,det]+data_err[order,det]*s2[order,det]
                ax1.fill_between(data_wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
                err = data_err[order,det] if np.all(np.isnan(data_err[order,det]))==False else 0
                errmean=np.nanmean(err*s2[order,det])
                if show_opacities==None:
                    if np.nansum(data_flux[order])!=0: # skip empty orders
                        ax2.errorbar(np.min(data_wave[order,det])-0.3, 0, yerr=errmean, 
                                    ecolor=retr_obj.color, elinewidth=1, capsize=2)
            
            if x==0 and det==0:
                ncol=2
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        #mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retr.color, linewidth=2,label=model_label)]
                if plot_components==True:
                    lines.append(Line2D([0], [0], color=prim_c,linewidth=2,label='A'))
                    lines.append(Line2D([0], [0], color=sec_c,linewidth=2,label='B'))
                    ncol+=2
                if plot_veiling:
                    lines.append(Line2D([0], [0], color=noveil_c,linewidth=2,label=noveil_label))
                    ncol+=1
                elif plot_cloud:
                    lines.append(Line2D([0], [0], color=nocloud_c,linewidth=2,label=nocloud_label))
                    ncol+=1
                leg=ax1.legend(handles=lines,fontsize=12,ncol=ncol,bbox_to_anchor=(0.47,1.4),loc='upper center')
                leg.get_frame().set_linewidth(0.0)

            if show_opacities==None:
                ax2.plot([np.min(data_wave[order,det]),np.max(data_wave[order,det])],[0,0],lw=0.8,c='k')
            else:
                opacities_legend=[]
                for i,species in enumerate(show_opacities):
                    opas=opacities[species]
                    ymax=np.nanmax([np.nanmax(model_flux[order]),np.max(data_flux[order])])#*0.9
                    ymin=np.nanmin([np.nanmin(model_flux[order]),np.min(data_flux[order])])#*1.1
                    if f'log_{species}' in retr_obj.params_dict:
                        opa=opas[order,det]*10**retr_obj.params_dict[f'log_{species}']
                        y_cutoff = 1e-8
                        tick_spacing=1
                        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
                    else: 
                        opa=opas[order,det]
                        y_cutoff = 1e-1 if np.max(opa)>1e-1 else np.min(opa)
                    for a in [ax1,ax2]:
                        a.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                        a.grid(which='minor', axis='x', linestyle=':', linewidth=0.6,alpha=0.9)
                        a.grid(which='major', axis='x', linestyle='-', linewidth=0.6,alpha=1)
                        a.tick_params(axis='x', which='minor', bottom=False)

                    opa_scaled=scale_between(ymax,ymin,opa)
                    opa_orders.append(opa)
                    cl = retr_obj.species_info.loc[species,'color']
                    ax2.plot(data_wave[order,det],opa,lw=0.8,c=cl)
                    #ax1.plot(data_wave[order,det],opa_scaled,lw=0.8,c=opacities_colors[i])
                    opacities_legend.append(Line2D([0],[0],color=cl,
                                                   linewidth=2,linestyle='-',label=species))
                if idx==0 and det==0:
                    ax2.legend(handles=opacities_legend,fontsize=10,ncol=len(show_opacities))

        if np.isnan(data_flux[order]).all():
            #if 'log_k_rk' in retr.params_dict:
                #min_array.append(np.nanmin([data_flux,model_flux,disk_flux]))
                #max_array.append(np.nanmax([data_flux,model_flux,disk_flux]))
            #else:
            min_array.append(np.nanmin([data_flux,model_flux]))
            max_array.append(np.nanmax([data_flux,model_flux]))
        else:
            min_array.append(np.nanmin([data_flux[order],model_flux[order]]))
            max_array.append(np.nanmax([data_flux[order],model_flux[order]]))

        #min1=np.nanmin(np.array([retr.data_flux[order]-retr.data_err[order],retr.model_flux[order]]))
        #max1=np.nanmax(np.array([retr.data_flux[order]+retr.data_err[order],retr.model_flux[order]]))
        min1=np.nanmin(np.array(min_array))
        max1=np.nanmax(np.array(max_array))
        ax1.set_ylim(min1,max1)
        if np.nansum(residuals[order])!=0 and show_opacities==None:
            ax2.set_ylim(np.nanmin(residuals[order]),np.nanmax(residuals[order]))
        elif show_opacities==None:# if empty order full of nans
            ax2.set_ylim(-0.1,0.1)
        else:
            ax2.set_xlim(np.min(data_wave[order]),np.max(data_wave[order]))
            ax1.set_xlim(np.min(data_wave[order]),np.max(data_wave[order]))
            ax2.set_yscale('log')
            if np.min(opa_orders)<y_cutoff:
                ax2.set_ylim(y_cutoff,np.max(opa_orders))
        ax1.tick_params(labelbottom=False)  # don't put tick labels at bottom
        ax1.tick_params(axis="both")
        ax2.tick_params(axis="both")
        ax1.set_ylabel('Normalized Flux')
        if show_opacities==None:
            ax2.set_ylabel('Res.')
        else:
            ax2.set_ylabel('[g/cm$^2$]')
        ax1.tick_params(labelsize=9)
        ax2.tick_params(labelsize=9)
        if x!=(n_sub-1):
            ax3.set_visible(False) # invisible for spacing
        x+=3
    ax[(n_sub-2)].set_xlabel('Wavelength [nm]')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    if show_opacities!=None:
        name = f'bestfit_{species_string}'
    elif plot_veiling:
        name = 'bestfit_veiling' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_veiling'
    elif plot_cloud:
        name = 'bestfit_cloud' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_cloud'
    elif plot_components==False:
        name = 'bestfit_split' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_split'
    else:
        name = 'bestfit_components' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}components'
    fig.savefig(f'{retr_obj.output_dir}/{name}.pdf')
    plt.close()

def plot_pt(retr_obj,fs=12,figsize=5,sb=True,show_cond=False,show_contr=True,**kwargs):

    legend_labels=kwargs.get('legend_labels',None)

    if show_cond: # show condensation curve
        if retr_obj.chemistry in ['equchem','quequchem']:
            C_O = retr_obj.model_object.params['C/O']
            Fe_H = retr_obj.model_object.params['Fe/H']
        elif retr_obj.chemistry =='flexequ':
            C_O = retr_obj.params_dict['C/O']
            Fe_H = retr_obj.params_dict['C/H']
        elif retr_obj.chemistry=='freechem':
            C_O = retr_obj.model_object.CO
            Fe_H = retr_obj.model_object.FeH   

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(figsize,figsize),dpi=200)
    #cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    #cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    #cs_colors=['gold','goldenrod','peru','sandybrown']
    cloud_species = ['MgSiO3(c)', 'Fe(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)']
    cs_colors=['goldenrod','sandybrown']

    # if pt profile and condensation curve don't intersect, clouds have no effect
    if retr_obj.target.name in ['2M0355','2M1425','test','test_corr','testsys'] and show_cond:
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
    if retr_obj.target.name in ['test','test_corr','testsys']:
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

    elif retr_obj.target.name in ['2M0355','2M1425'] and sb==True:
        file=np.loadtxt('t1600g562nc_m0.0.dat')
        pres=file[:,1] # bar
        temp=file[:,2] # K
        ax.plot(temp,pres,linestyle='dashdot',c='blueviolet',linewidth=2)
        comparison_pt=Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=1600\,$K, log$\,g=4.75$')
    #elif retr_obj.target.name=='ROXs12B' and sb==True:
        #file=np.loadtxt('t2400g562nc_m0.0.dat')
        #pres=file[:,1] # bar
        #temp=file[:,2] # K
        #ax.plot(temp,pres,linestyle='dashdot',c='blueviolet',linewidth=2)
        #comparison_pt=Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=2400\,$K, log$\,g=4.75$')

    if retr_obj.target.name=='2M0355': # compare with Zhang2022 science verification
        PT_Zhang=np.loadtxt(f'CRIRES/{retr_obj.target.name}/2M0355_PT_Zhang2021.dat')
        p_zhang=PT_Zhang[:,0]
        t_zhang=PT_Zhang[:,1]
        ax.plot(t_zhang,p_zhang,linestyle='dashdot',c='cornflowerblue',linewidth=2)

    if retr_obj.target.name in ['Sorg1X','Sorg20X']:
        tab = PSG_input(retr_obj.target.name)
        ax.plot(tab.temperature,tab.pressure,linestyle='dashdot',c='blueviolet',linewidth=2)
        comparison_pt=Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Input $P$-$T$')

    if 'retr_obj2' in kwargs: # if compare two retrs, specify object name in legend
        object_label=f'{retr_obj.target.name} $P$-$T$'
        contr_label=f'{retr_obj.target.name} contr.'
    else:
        object_label='$P$-$T$ profile'
        contr_label='Contribution'

    lines=[]
    # plot PT-profile + errors on retrieved temperatures
    def plot_temperature(retr_obj,ax,olabel): 
        if retr_obj.PT_type=='PTknot':
            #ax.plot(retr_obj.model_object.temperature,retr_obj.model_object.pressure,color=retr_obj.color,lw=2) 
            medians=[]
            errs=[]
            log_P_knots=retr_obj.model_object.log_P_knots
            t_keys = [key for key in retr_obj.params_dict.keys() if re.fullmatch(r"T\d+", key)] 
            t_keys = sorted(t_keys, key=lambda x: int(x[1:]))[::-1] # start at top of atmosphere, T0 last
            for key in t_keys: # order T4,T3,T2,T1,T0 like log_P_knots
                medians.append(retr_obj.params_dict[key])
                errs.append(retr_obj.params_dict[f'{key}_err'])
            errs=np.array(errs)
            for x in [1,2,3]: # plot 1-3 sigma errors
                lower = CubicSpline(log_P_knots,medians+x*errs[:,0])(np.log10(retr_obj.pressure))
                upper = CubicSpline(log_P_knots,medians+x*errs[:,1])(np.log10(retr_obj.pressure))
                ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color,alpha=0.15)
            ax.scatter(medians,10**retr_obj.model_object.log_P_knots,color=retr_obj.color)
            temp_median = CubicSpline(log_P_knots,medians)(np.log10(retr_obj.pressure))
            ax.plot(temp_median,retr_obj.model_object.pressure,color=retr_obj.color,lw=2) 

            xmin=np.min(lower)-20#-100
            xmax=np.max(upper)+20#+100
            lines.append(Line2D([0],[0],marker='o',color=retr_obj.color,markerfacecolor=retr_obj.color,
                    linewidth=2,linestyle='-',label=olabel))

        elif retr_obj.PT_type in ['PTgrad','PTgradvar']:
            dlnT_dlnP_knots=[]
            derr=[]
            n_grad = sum(1 for key in retr_obj.params_dict if re.fullmatch(r'dlnT_dlnP_\d+', key))
            for i in range(n_grad):
                key=f'dlnT_dlnP_{i}'
                dlnT_dlnP_knots.append(retr_obj.params_dict[key]) # gradient median values
                derr.append(retr_obj.params_dict[f'{key}_err']) # -/+ errors
            derr=np.array(derr) # gradient errors
            T0=retr_obj.params_dict['T0']
            err=retr_obj.params_dict['T0_err']
            temperature=retr_obj.model_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots,T_base=T0)
            ax.plot(temperature,retr_obj.model_object.pressure,color=retr_obj.color,lw=2) 
            # get 1-2-3 sigma of temp_dist, has shape (samples, n_atm_layers)
            quantiles = np.array([np.percentile(retr_obj.temp_dist[:,i], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1) for i in range(retr_obj.temp_dist.shape[1])])
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,0],quantiles[:,-1],color=retr_obj.color,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,1],quantiles[:,-2],color=retr_obj.color,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,2],quantiles[:,-3],color=retr_obj.color,alpha=0.15)
            xmin=np.min((quantiles[:,0],quantiles[:,-1]))-100
            xmax=np.max((quantiles[:,0],quantiles[:,-1]))+100
            lines.append(Line2D([0], [0], color=retr_obj.color,
                                linewidth=2,linestyle='-',label=olabel))
            if retr_obj.PT_type=='PTgradvar':
                interp_func = interp1d(np.log10(retr_obj.pressure), retr_obj.model_object.temperature, 
                                        kind='linear', fill_value="extrapolate")
                T_knots = interp_func(retr_obj.model_object.log_P_knots)
                ax.scatter(T_knots,10**retr_obj.model_object.log_P_knots,
                            c=retr_obj.color,s=20)
        if retr_obj.PT_type=='PTguillot':
            T_int = retr_obj.params_dict['T_int']
            T_equ = retr_obj.params_dict['T_equ']
            kappa_IR = 10**retr_obj.params_dict['log_k_IR']
            gamma = 10**retr_obj.params_dict['log_gamma']
            gravity = 10**retr_obj.params_dict['log_g']
            temperature=retr_obj.model_object.make_pt()
            ax.plot(temperature,retr_obj.model_object.pressure,color=retr_obj.color,lw=2) 
            # get 1-2-3 sigma of temp_dist, has shape (samples, n_atm_layers)
            quantiles = np.array([np.percentile(retr_obj.temp_dist[:,i], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1) for i in range(retr_obj.temp_dist.shape[1])])
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,0],quantiles[:,-1],color=retr_obj.color,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,1],quantiles[:,-2],color=retr_obj.color,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,2],quantiles[:,-3],color=retr_obj.color,alpha=0.15)
            xmin=np.min((quantiles[:,0],quantiles[:,-1]))-100
            xmax=np.max((quantiles[:,0],quantiles[:,-1]))+100
            lines.append(Line2D([0], [0], color=retr_obj.color,
                                linewidth=2,linestyle='-',label=olabel))

        return xmin,xmax

    xmin,xmax=plot_temperature(retr_obj,ax,object_label)
       
    if retr_obj.target.name=='2M0355':
        lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    
    if 'retr_obj2' in kwargs: # compare two retrs
        retr_obj2=kwargs.get('retr_obj2')
        object_label2=f'{retr_obj2.target.name} $P$-$T$'
        xmin2,xmax2=plot_temperature(retr_obj2,ax,object_label2)
        xmin=np.nanmin([xmin,xmin2])
        xmax=np.nanmax([xmax,xmax2])
        if show_contr:
            summed_contr2=retr_obj2.summed_emcont
            contribution_plot2=summed_contr2/np.max(summed_contr2)*(xmax-xmin)+xmin
            ax.plot(contribution_plot2,retr_obj2.model_object.pressure,linestyle='dashed',
                    lw=1.5,alpha=0.8,color=retr_obj2.color)
            lines.append(Line2D([0], [0], color=retr_obj2.color, alpha=0.8,linewidth=1.5, 
                                linestyle='--',label=f'{retr_obj2.target.name} contr.'))
    
    if 'retr_obj3' in kwargs:
        retr_obj3=kwargs.get('retr_obj3')
        object_label3=f'{retr_obj3.target.name} retr'
        xmin3,xmax3=plot_temperature(retr_obj3,ax,object_label3)
        xmin=np.nanmin([xmin,xmin2,xmin3])
        xmax=np.nanmax([xmax,xmax2,xmax3])
        if show_contr:
            summed_contr3=retr_obj3.summed_emcont
            contribution_plot3=summed_contr3/np.max(summed_contr3)*(xmax-xmin)+xmin
            ax.plot(contribution_plot3,retr_obj3.model_object.pressure,linestyle='dashed',
                    lw=1.5,alpha=0.8,color=retr_obj3.color)
            lines.append(Line2D([0], [0], color=retr_obj3.color, alpha=0.8,linewidth=1.5, 
                                linestyle='--',label=f'{retr_obj3.target.name} contr.'))

    if show_contr:
        summed_contr=retr_obj.summed_emcont    
        contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
        ax.plot(contribution_plot,retr_obj.model_object.pressure,linestyle='dashed',
                lw=1.5,alpha=0.8,color=retr_obj.color)
        lines.append(Line2D([0], [0], color=retr_obj.color, alpha=0.8,
                            linewidth=1.5, linestyle='--',label=contr_label))
        lines = lines[:1]+[lines[-1]]+lines[1:-1] # move to second position in legend instead of last

        if retr_obj.target.name in ['Sorg1X','Sorg20X']:
            psg_contr=np.genfromtxt(f'LIFE/{retr_obj.target.name}/{retr_obj.target.name}_contr.txt',skip_header=1,delimiter=' ')
            psg_contr=psg_contr[:,1:] # exclude first column (wavelength)
            psg_contr=np.sum(psg_contr,axis=0)[::-1] # sum over all wavelengths, change order
            psg_contr =psg_contr/np.max(psg_contr)*(xmax-xmin)+xmin
            ax.plot(psg_contr,PSG_input(retr_obj.target.name).pressure,linestyle='dotted',
                    lw=1.5,alpha=0.8,color='blueviolet')
            lines.append(Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dotted',label='Input contr'))
    
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retr_obj.model_object.pressure),
        np.nanmin(retr_obj.model_object.pressure)),xlim=(xmin,xmax))
    
    if 'comparison_pt' in locals():
        lines.append(comparison_pt)

    if legend_labels!=None:
        lines=[]
        retr_objects=[retr_obj,retr_obj2]
        if 'retr_obj3' in kwargs:
            retr_objects.append(retr_obj3)
        for r,l in zip(retr_objects,legend_labels):
            lines.append(Line2D([0], [0], color=r.color,linewidth=2,linestyle='-',label=l))
            #lines.append(Line2D([0], [0], color=r.color, alpha=0.8,linewidth=1.5, linestyle='--',label=f'{l} cont.'))
                
    ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        name = 'PT_profile' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}PT_profile'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf')
        plt.close()

def get_plotposterior_labels(retr_obj,param_names,get_medians=False):
    param_labels = [] # mathtext  
    medians = []
    plot_posterior=np.empty((len(param_names),len(retr_obj.posterior[list(retr_obj.posterior.keys())[0]][0]))).T
    for i,key in enumerate(param_names):
        species_i = key[4:] #without log
        if species_i in retr_obj.vary_species: # posterior at maximum emission contribution
            log10_vmr_at_maxemcont = np.log10(np.array(retr_obj.VMR_dict[species_i])[:,retr_obj.params_dict['idx_maxcont']])
            plot_posterior[:,i]=log10_vmr_at_maxemcont
            param_labels.append(rf"log {retr_obj.species_info.loc[species_i,'mathtext_name']}")
            medians.append(np.nanmedian(log10_vmr_at_maxemcont))
        else:
            param_labels.append(retr_obj.posterior[key][1]) 
            plot_posterior[:,i]=np.array(retr_obj.posterior[key][0]).T
            medians.append((retr_obj.get_quantiles(retr_obj.posterior[key][0]))[0])
    if get_medians==True:
        return plot_posterior, param_labels, medians
    else:
        return plot_posterior, param_labels

def cornerplot(retr_obj,getfig=False,figsize=20,fs=12,plot_label='',alphas=False,
            only_abundances=False,only_params=None,not_abundances=False,ratios=False,
            cloud_params=False):
    
    param_names = []
    if only_abundances==True: # plot only abundances
        plot_label='_abunds'
        if retr_obj.chemistry in ['freechem','varchem']:
            #suffix='_0' if retr_obj.chemistry=='varchem' else ''
            abunds=[]
            param_names=[]
            species=retr_obj.species_names
            for spec in species:
                if spec in retr_obj.vary_species:
                    median_at_maxemcont = np.nanmedian(np.array(retr_obj.VMR_dict[spec])[:,retr_obj.params_dict['idx_maxcont']])
                    abunds.append(median_at_maxemcont)
                else:
                    #suffix='_0' if spec in retr_obj.vary_species else ''
                    abunds.append(retr_obj.params_dict[f'log_{spec}'])
                param_names.append(f'log_{spec}')

            abunds, param_names = zip(*sorted(zip(abunds, param_names)))
            param_names = param_names[::-1] # sort from most to least abundant

    elif only_params is not None: # keys of specified parameters to plot
        param_names = only_params
        plot_label='_some'

    elif not_abundances==True: # plot all except abundances
        plot_label='_rest'
        set_diff = np.setdiff1d(list(retr_obj.parameters.free_params.keys()),[f'log_{s}' for s in retr_obj.species_names])
        for key in set_diff:
            if not re.search(r'_\d+', key): # not include rest of abundances for varchem
                param_names.append(key)

    elif ratios==True:
        figsize=9
        plot_label='_ratios'
        param_names = get_ratios(retr_obj)

    elif alphas==True: # only for flexequ, scaling factors of equchem
        plot_label='_alphas'
        param_names=[]
        species=retr_obj.species_names
        for spec in species:
            if spec not in ['13CO','C17O','C18O','H2(18)O']:
                param_names.append(f"log_a_{spec}")

    elif cloud_params:
        plot_label='_cloud'
        figsize=6
        param_names=['log_opa_base_gray','log_P_base_gray','fsed_gray']
        if 'cloud_slope' in retr_obj.parameters.params:
            param_names.append('cloud_slope')

    else: # all params: avoid, could be too big
        plot_label='_all'
        param_names= list(retr_obj.parameters.free_params.keys())

    plot_posterior, param_labels, medians = get_plotposterior_labels(retr_obj,param_names,get_medians=True)
    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues
    fig = corner.corner(plot_posterior, 
                        labels=param_labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retr_obj.color,
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

    #corner.overplot_lines(fig,medians,color=retr_obj.color,lw=1.3,linestyle='solid') # plot median values of posterior

    # add true values of test spectrum, plotting didn't work bc x-axis range so small, some didn't show up
    if retr_obj.target.name in ['test','test_corr','testsys']:
        from testspec import test_parameters,test_mathtext
        compare=[]
        for key_i in param_names:
            if key_i in test_parameters.keys():
                label_i = test_mathtext[key_i]
                value_i=test_parameters[key_i]
                compare.append(value_i) # add only those values that are used in cornerplot
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'in: {compare[i]}'
            fig.axes[x].title.set_text(titles[x])
            x+=len(param_labels)+1

        # Adjust tick label font size
        axes = fig.get_axes()
        for ax in axes:
            ax.tick_params(axis="both", labelsize=fs*0.5)  # Tick label font size

    if "Sorg" in retr_obj.target.name and only_abundances==True:
        # input VMRs at input's maximum emission contr
        VMR_maxcont = load_pickle(f'LIFE/{retr_obj.target.name}/VMR_maxcont.pickle')

        # maximum VMR of each species in inpu
        # emission from some species come from higher altitudes
        VMR_maxtot = load_pickle(f'LIFE/{retr_obj.target.name}/VMR_maxtotal.pickle') 
        sorted_keys = [key for key in param_names if key in VMR_maxcont] # so that it's at correct position
        sorted_keys = [key for key in param_names if key in VMR_maxtot] # so that it's at correct position
        
        comp_maxcont=[]
        comp_maxtot=[]
        for key_i in sorted_keys:
            maxcont=VMR_maxcont[key_i]
            maxtot = VMR_maxtot[key_i]
            if key_i in param_names:
                comp_maxcont.append(maxcont) # add only those values that are used in cornerplot
                comp_maxtot.append(maxtot)
        x=0
        for i in range(len(comp_maxcont)):
            titles[x] = titles[x]+'\n'+f'em: {np.round(comp_maxcont[i],decimals=2)}'+'\n'+f'max: {np.round(comp_maxtot[i],decimals=2)}'
            fig.axes[x].title.set_text(titles[x])
            x+=len(param_labels)+1

        # Adjust tick label font size
        axes = fig.get_axes()
        for ax in axes:
            ax.tick_params(axis="both", labelsize=fs*0.5)  # Tick label font size
    
    elif "Sorg" in retr_obj.target.name:
        #from LIFE.Sorg20X.LIFE_pRT import LIFE_parameters
        from LIFE_pRT import LIFE_parameters
        compare=[]
        for key_i in param_names:
            if key_i in LIFE_parameters.keys():
                value_i=LIFE_parameters[key_i]
                compare.append(value_i) # add only those values that are used in cornerplot
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'in: {np.round(compare[i],decimals=2)}'
            fig.axes[x].title.set_text(titles[x])
            fig.axes[x].axvline(compare[i],c='r',lw=0.8,ls='dashed')
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
    if retr_obj.instrument=='CRIRES':
        plot_spectrum_split(retr_obj)
        plot_spectrum_inset(retr_obj)
        plot_pt(retr_obj)
        if 'log_k_rk' in retr_obj.params_dict:
            plot_rk(retr_obj)
        if 'cloud_slope' in retr_obj.params_dict:
            cornerplot(retr_obj,cloud_params=True)
        comp_equ=True # compare with what equchem abundances would be like
        if retr_obj.primary_label==False: 
            plot_spectrum_split(retr_obj,plot_components=True)
            if 'log_phi_k' in retr_obj.parameters.params.keys(): # if linear func
                plot_phi_components(retr_obj)
    elif retr_obj.instrument=='LIFE':
        plot_spectrum_inset(retr_obj,inset=False)
        plot_pt(retr_obj,show_cond=False)
        comp_equ=False
    summary_plot(retr_obj)
    opacity_plot(retr_obj)
    if retr_obj.chemistry in ['freechem','varchem']:
        cornerplot(retr_obj,ratios=True) # plot ratios, already in equchem cornerplot by default
        if split_corner: # split corner plot to avoid massive files
            cornerplot(retr_obj,only_abundances=True)
            cornerplot(retr_obj,not_abundances=True)
        else: # make cornerplot with all parameters, could be huge, avoid this
            cornerplot(retr_obj,only_params=only_params)
    elif retr_obj.chemistry in ['equchem','quequchem','flexequ']:
        comp_equ=False
        if split_corner: # split corner plot to avoid massive files
            if retr_obj.chemistry in ['equchem','quequchem']:
                only_params=['rv','vsini','log_g']
                if {'Fe/H', 'C/O'}.issubset(retr_obj.parameters.free_params):
                    only_params.extend(['C/O', 'Fe/H'])
            elif retr_obj.chemistry =='flexequ':
                only_params=['rv','vsini','log_g']
            if '13CO' in retr_obj.species_names:
                only_params.append('log_C12_13_ratio')
            if 'H2(18)O' in retr_obj.species_names:
                only_params.append('log_H2O16_18_ratio')
            if 'C18O' in retr_obj.species_names:
                only_params.append('log_O16_18_ratio')
            if 'C17O' in retr_obj.species_names:
                only_params.append('log_O16_17_ratio')
            if retr_obj.chemistry=='quequchem':
                for val in ['log_Pqu_CO_CH4','log_Pqu_NH3','log_Pqu_HCN']:
                    only_params.append(val)
            cornerplot(retr_obj,only_params=only_params,plot_label='1')
            only_params2=list(set(retr_obj.parameters.param_keys)-set(only_params))
            cornerplot(retr_obj,only_params=only_params2,plot_label='2')
        else: # avoid this though
            cornerplot(retr_obj,only_params=only_params)
        if retr_obj.chemistry=='flexequ':
            plot_scaled_abunds(retr_obj)
            cornerplot(retr_obj,ratios=True)
    VMR_plot(retr_obj,VMR_species='all',comp_equ=comp_equ) # show all (without errors)
    VMR_plot(retr_obj,comp_equ=comp_equ) # show most abundant (with errors)
    if retr_obj.primary_label==False:
        plot_spectrum_split(retr_obj,plot_components=True)
    
def summary_plot(retr_obj,**kwargs):

    fs=13
    figsize=17

    if retr_obj.instrument=='LIFE':

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_axes([0.1, 0.55, 0.2, 0.2])  # Top-left
        l, b, w, h = [0.45, 0.47, 0.55, 0.17] # left, bottom, width, height
        ax2 = fig.add_axes([l,b,w,h])   # Top-right
        ax3 = fig.add_axes([0.1, 0.1, 0.25, 0.4])    # Bottom-left
        ax4 = fig.add_axes([0.45, 0.1, 0.55, 0.3]) # Bottom-right
        ax5 = fig.add_axes([0.45, 0.64, 0.55, 0.15]) 

        param_names= ['log_g']
        plot_posterior, param_labels, medians = get_plotposterior_labels(retr_obj,param_names,get_medians=True)
        ax1.hist(plot_posterior, bins=20, color=retr_obj.color)
        ax1.set_yticks([])
        minus_err, plus_err = np.percentile(plot_posterior, [16.0,84.0])
        title = f'log$g$ = {np.round(medians[0],decimals=2)}$^{{+{np.round(plus_err-medians[0],decimals=2)}}}_{{{np.round(minus_err-medians[0],decimals=2)}}}$'
        ax1.set_title(f'{title} \nin: 3.09',fontsize=fs)
        ax1.tick_params(labelsize=fs)

        ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
        plot_spectrum_inset(retr_obj,ax=(ax2,ax_res),inset=False,fs=fs)
        ax_res.cla()
        ax_res.axis('off')
        ax2.set_xlabel('Wavelength [$\mu$m]',fontsize=fs)
        ax2.set_ylabel('photons s$^{-1}$ m$^{-3}$',fontsize=fs)

        plot_pt(retr_obj,ax=ax3,fs=fs*0.8)
        plot_species = retr_obj.species_names.copy()
        plot_species.append('H2')
        plot_species.append('He')
        VMR_plot(retr_obj,fs=fs,ax=ax4,VMR_species=plot_species,
                xmin=1e-8,xmax=1e-0,plotlegend=True)
        del plot_species

        opacity_plot(retr_obj,n=10,ax=ax5,fs=fs)
        ax5.set_xticks([])
        ax5.set_xlabel('')

        name = 'summary' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}summary'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
        return

    if retr_obj.chemistry in ['equchem','quequchem','flexequ']:
        if retr_obj.chemistry in ['equchem','quequchem']:
            only_params=['rv','vsini','log_g']
            if {'Fe/H', 'C/O'}.issubset(retr_obj.parameters.free_params):
                only_params.extend(['C/O', 'Fe/H'])
        elif retr_obj.chemistry =='flexequ':
            only_params=['rv','vsini','log_g']
        if '13CO' in retr_obj.species_names:
            only_params.append('log_C12_13_ratio')
        if 'C18O' in retr_obj.species_names:
            only_params.append('log_O16_18_ratio')
        if 'H2(18)O' in retr_obj.species_names:
            only_params.append('log_H2O16_18_ratio')
        if 'C17O' in retr_obj.species_names:
            only_params.append('log_O16_17_ratio')
    elif retr_obj.chemistry in ['freechem','varchem']:
        only_params=['rv','vsini','log_g']
        if retr_obj.instrument=='LIFE':
            only_params=['log_g']
        abunds=[]
        param_names=[]
        species=retr_obj.species_names
        #suffix='_0' if retr_obj.chemistry=='varchem' else ''
        for spec in species:
            suffix='_0' if spec in retr_obj.vary_species else ''
            abunds.append(retr_obj.params_dict[f'log_{spec}{suffix}'])
            param_names.append(f"log_{spec}{suffix}")
        abunds, param_names = zip(*sorted(zip(abunds, param_names)))
        only_params.extend(param_names[-8:][::-1]) # get most abundant species
    if 'show_params' in kwargs:
        only_params=kwargs.get('show_params')
        figsize=15
        fs=9

    fig, ax = cornerplot(retr_obj,getfig=True,only_params=only_params,figsize=figsize,fs=fs)
    l, b, w, h = [0.42,0.84,0.55,0.15]#[0.37,0.84,0.6,0.15] # left, bottom, width, height
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

    l1 = opacity_plot(retr_obj,n=n1,ax=axes[0],smallrange=True,addname=True)
    l2= opacity_plot(retr_obj2,n=n2,ax=axes[1],smallrange=True,addname=True)
    merged = list(set(l1).union(l2))
    colors=[]
    labels=[]
    for col,lab in merged:
        colors.append(col)
        labels.append(lab)
    show_species = ['H2O','12CO','13CO','HF','H2S','CH4','H2(18)O','NH3'] # legend should be in this order
    mathtext=[]
    for spec in show_species:
        mathtext.append(retr_obj.species_info.loc[spec,'mathtext_name'])
    labels, colors = list(zip(*sorted(zip(labels,colors), key=lambda x: mathtext.index(x[0]))))
    lines=[]
    for col,lab in zip(colors,labels):
        lines.append(Line2D([0],[0],color=col,linewidth=2,label=lab))
    legend=axes[0].legend(handles=lines,ncol=(len(lines)//2+len(lines)%2),loc='lower center', bbox_to_anchor=(0.47, 1.0))
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((0, 0, 0, 0))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(f'{retr_obj.output_dir}/comparison/opacities_both.pdf',bbox_inches="tight",dpi=200)
    plt.close()

def opacity_plot(retr_obj,only_params=None,n=7,fs=10,smallrange=False,addname=False,**kwargs): # n most abundant species

    only_params=[]
    abunds=[]
    pRT_names=[]
    labels=[]
    species=retr_obj.species_names

    if retr_obj.chemistry in ['freechem','varchem']:
        #suffix='_0' if retr_obj.chemistry=='varchem' else ''
        for spec in species:
            if spec in retr_obj.vary_species and spec not in ['H2','He']:
                median_at_maxemcont = np.nanmedian(np.array(retr_obj.VMR_dict[spec])[:,retr_obj.params_dict['idx_maxcont']])
                abunds.append(median_at_maxemcont)
            #suffix='_0' if spec in retr_obj.vary_species else ''
            elif spec not in ['H2','He']:
                #abunds.append(retr_obj.params_dict[f"log_{spec}{suffix}"])
                abunds.append(retr_obj.params_dict[f"log_{spec}"])
        
    elif retr_obj.chemistry in ['equchem','quequchem','flexequ']: # use VMRs where emission contribution is maximal
        for spec in species:
            abunds.append(np.median(retr_obj.VMR_dict[spec],axis=0)[find_nearest(retr_obj.pressure,10**retr_obj.params_dict['log_P_maxcont'])])
                
    abunds, species = zip(*sorted(zip(abunds, species)))
    only_params=species[-n:][::-1] # get largest n
    abunds = abunds[-n:][::-1] # get largest n
    VMRs=[]
    colors=[]
    for i,par in enumerate(only_params):
        pRT_names.append(retr_obj.species_info.loc[par,'pRT_name'])
        labels.append(retr_obj.species_info.loc[par,'mathtext_name'])
        colors.append(retr_obj.species_info.loc[only_params[i],'color'])
        if retr_obj.chemistry in ['freechem','varchem']:
            if only_params[i] in retr_obj.vary_species:
                median_at_maxemcont = np.nanmedian(np.array(retr_obj.VMR_dict[only_params[i]])[:,retr_obj.params_dict['idx_maxcont']])
                VMRs.append(median_at_maxemcont)
                #VMRs.append(10**retr_obj.params_dict[f"log_{only_params[i]}_0"])
            else:
                VMRs.append(10**retr_obj.params_dict[f"log_{only_params[i]}"])
        elif retr_obj.chemistry in ['equchem','quequchem','flexequ']:
            VMRs.append(abunds[i])

    if retr_obj.instrument=='CRIRES':
        wl_unit='nm'
        Kband=retr_obj.target.K2166
        wlen_range=np.array([np.min(Kband),np.max(Kband)])*1e-3 # nm to microns
        atmosphere = Radtrans(line_species=pRT_names,
                        rayleigh_species = ['H2', 'He'],
                        continuum_opacities = ['H2-H2', 'H2-He'],
                        wlen_bords_micron=wlen_range, 
                        mode='lbl',
                        lbl_opacity_sampling=10)
    elif retr_obj.instrument=='LIFE':
        wl_unit=r'\mathrm{\mu}m'
        wlen_range=np.array([np.min(retr_obj.data_wave),np.max(retr_obj.data_wave)]) # in microns
        atmosphere = Radtrans(line_species=pRT_names,
                        rayleigh_species = ['H2', 'He'],
                        continuum_opacities = ['H2-H2', 'H2-He'],
                        wlen_bords_micron=wlen_range, 
                        mode='c-k')

    # use temperature at maximum contribution
    wave_cm, opas = atmosphere.get_opa(np.array([retr_obj.params_dict['T_maxcont']]).reshape(1))
    wave_nm = wave_cm*1e7
    wave_um = wave_cm*1e4
    wave_plot = wave_nm if retr_obj.instrument=='CRIRES' else wave_um
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
        spec,=ax.plot(wave_plot,opas[m]*VMRs[i],lw=0.5,c=colors[i])
        lines.append(Line2D([0],[0],color=spec.get_color(),
                        linewidth=2,label=labels[i]))
        line_props.append((spec.get_color(),labels[i]))

    if addname==True:
        from matplotlib import colors
        fc=colors.to_rgba(retr_obj.color)
        fc = fc[:-1] + (0.5,) # <--- Change the alpha value of facecolor to be 0.7
        ax.annotate(retr_obj.target.name, xy=(0.5, 0.88), xycoords='axes fraction',
                    ha='center', va='center', c='k', fontsize=12, 
                    bbox={'boxstyle':'round', 'fc':fc, 'ec':'k'})
        
    if retr_obj.instrument=='CRIRES':
        for order in range(7):
            for det in range(3):
                ax.fill_betweenx([ymin,ymax],Kband[order,det][0],Kband[order,det][1],color='k',alpha=0.063)
                ax.set_xlim(np.min(Kband),np.max(Kband))
    elif retr_obj.instrument=='LIFE':
        ax.set_xlim(np.min(wlen_range),np.max(wlen_range))
    ax.set_yscale('log')
    ax.set_ylabel('Opacity [cm$^2$/g]',fontsize=fs)
    ax.set_xlabel(f"Wavelength [{wl_unit}]",fontsize=fs)

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

def compare_retrievals(retr_obj1,retr_obj2,fs=12,with_pt=True,ratios_logg=False,figsize=12,**kwargs):

    legend_labels=kwargs.get('legend_labels',None)
    num=2 # number of retrs
    suffix=''
    l, b, w, h = [0.58,0.65,0.39,0.39] # left, bottom, width, height for PT plot

    # can only compare freechem+freechem or freechem+equchem/quequchem(+equchem/quequchem)
    if retr_obj1.chemistry=='freechem' and retr_obj2.chemistry=='freechem':

        if 'show_params' in kwargs:
            only_params=kwargs.get('show_params')
        elif ratios_logg==True:
            suffix = 'ratios_'
            with_pt=False
            figsize=9
            only_params= ['log_g','C/O','C/H','log_12CO/13CO','log_H2O/H2(18)O']
        else:
            suffix = 'abunds_'
            only_params=['log_H2O','log_12CO','log_13CO','log_CH4','log_H2S','log_HF','log_H2(18)O','log_NH3']

        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params)
        posterior2, _ = get_plotposterior_labels(retr_obj2,only_params)

    elif retr_obj1.chemistry=='freechem' and retr_obj2.chemistry in ['equchem','quequchem','flexequ']:

        suffix='chems'
        #only_params=['log_g','C/O','C/H','log_12CO/13CO','log_12CO/C17O','log_12CO/C18O','log_H2O/H2(18)O']
        only_params=['log_g','C/O','C/H','log_12CO/13CO','log_H2O/H2(18)O']

        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params)

        # add log_O16_18_ratio to equchem again, bc freechem has C18O and H218O ratios
        #only_params=['log_g','C/O','Fe/H','log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_O16_18_ratio']
        only_params=['log_g','C/O','Fe/H','log_C12_13_ratio','log_H2O16_18_ratio']
        posterior2, labels = get_plotposterior_labels(retr_obj2,only_params)
        figsize=8.5
        legend_labels = ['Free','Equ']
        l, b, w, h = [0.62,0.72,0.33,0.33] # left, bottom, width, height for PT plot

        if 'retr_obj3' in kwargs: # quequchem
            num=3
            retr_obj3=kwargs.get('retr_obj3')
            posterior3, labels = get_plotposterior_labels(retr_obj3,only_params)
            legend_labels.append('Quequ')

    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues
    
    def plot_corner(posterior,retr_obj,labels,fig,getfig=False):
        ranges = [(np.min([np.min(posterior1[:, i]),np.min(posterior2[:, i])]), 
                   np.max([np.max(posterior1[:, i]),np.max(posterior2[:, i])])) for i in range(posterior1.shape[1])]
        if retr_obj.target.name=='2M1425' and suffix =='abunds_':
            ranges[3] = (-11,-4.7) # increase range for CH4 to make better visible
            ranges[4] = (-11,-3.5) # increase range for H2S to make better visible
            ranges[6] = (-12,-4.7) # increase range for H218O to make better visible
        fig = corner.corner(posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize': fs},
                        label_kwargs={'fontsize': fs*0.8},
                        color=retr_obj.color,
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
    colors_list=[retr_obj1.color,retr_obj2.color]

    if 'retr_obj3' in kwargs:
        titles3=plot_corner(posterior3,retr_obj3,labels,fig)
        enum=[0,1,2]
        titles_list.append(titles3)
        colors_list.append(retr_obj3.color)

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
            if run == 0: # first retr, add parameter name
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
        
        ax_PT = fig.add_axes([l,b,w,h])
        if 'retr_obj3' not in kwargs:
            plot_pt(retr_obj1,retr_obj2=retr_obj2,ax=ax_PT,legend_labels=legend_labels,sb=False)
        else:
            plot_pt(retr_obj1,retr_obj2=retr_obj2,
                    ax=ax_PT,retr_obj3=retr_obj3,legend_labels=legend_labels,sb=False)

    comparison_dir=pathlib.Path(f'{retr_obj1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(f'{comparison_dir}/cornerplot_{suffix}{num}.pdf',bbox_inches="tight",dpi=200)
    plt.close()

def double_VMR_plot(name1,name2, **kwargs):
    from config_run import init_retrieval

    if 'show_species' in kwargs:
        show_species=kwargs.get('show_species')
    else:
        show_species=['H2O','12CO','13CO','CH4','H2S','NH3','HF','HCN']

    # 2 VMR plots of 2 object comparing free vs equ chem
    fig,axes=plt.subplots(2,1,figsize=(5,6),dpi=200)

    def VMRs_free_equ(obj_name,ax):
        free=init_retrieval(target=obj_name,PT_type='PTgrad',chem='freechem',Nlive=400,evtol=0.5)
        free.evaluate(makefigs=False)
        equ=init_retrieval(target=obj_name,PT_type='PTgrad',chem='equchem',Nlive=400,evtol=0.5)
        equ.evaluate(makefigs=False)
        legend_elements = VMR_plot(retr_obj=free,retr_obj2=equ,VMR_species=show_species,ax=ax,addname=True)
        return legend_elements, free

    l1, retr_obj = VMRs_free_equ(name1,ax=axes[0])
    l2, _ = VMRs_free_equ(name2,ax=axes[1])
    merged = list(set(l1).union(l2))
    colors=[]
    labels=[]
    for col,lab in merged:
        colors.append(col)
        labels.append(lab)
    mathtext=[]
    for spec in show_species:
        mathtext.append(retr_obj.species_info.loc[spec,'mathtext_name'])
    labels, colors = list(zip(*sorted(zip(labels,colors), key=lambda x: mathtext.index(x[0]))))
    lines=[]
    for col,lab in zip(colors,labels):
        lines.append(Line2D([0],[0],color=col,linewidth=2,label=lab))
    # legend should be in order of species
    legend=axes[0].legend(handles=lines,ncol=(len(lines)//2+len(lines)%2),loc='lower center', 
                            bbox_to_anchor=(0.48, 1.0),fontsize=9)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((0, 0, 0, 0))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    axes[0].xaxis.set_ticks_position('top')
    axes[0].set_xticklabels([])  # Remove x tick labels
    yticklabels = axes[1].get_yticklabels()
    yticklabels[1].set_visible(False) # to avoid overlap
    fig.savefig(f'{retr_obj.output_dir}/comparison/VMRs_both.pdf',bbox_inches="tight",dpi=200)
    plt.close()

def VMR_plot(retr_obj,fs=10,n=8,VMR_species=None,comp_equ=False,
                addname=False,plotlegend=False,wH2He=False,
                xmin=1e-10,xmax=1e-1,**kwargs):

    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    suffix=''
    output_dir=retr_obj.output_dir

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(5,3.5),dpi=200)

    alpha=0.6 if 'retr_obj2' in kwargs or comp_equ==True else 1
    legend_labels=0
    #xmin,xmax=1e-10,10**(-2.5)
    chemleg=[] # legend for chemistry
    pressure=retr_obj.model_object.pressure

    # plot n most abundant species
    if VMR_species==None:
        suffix='_few'
        abunds=[]
        species=retr_obj.species_names
        
        if retr_obj.chemistry in ['freechem','varchem']:
            #s='_0' if retr_obj.chemistry =='varchem' else ''
            for spec in species:
                s='_0' if spec in retr_obj.vary_species else ''
                abunds.append(retr_obj.params_dict[f"log_{spec}{s}"])
        
        elif retr_obj.chemistry in ['equchem','quequchem','flexequ']: # use VMRs where emission contribution is maximal
            for spec in species:
                abunds.append(np.median(retr_obj.VMR_dict[spec],axis=0)[find_nearest(retr_obj.pressure,10**retr_obj.params_dict['log_P_maxcont'])])

        abunds, species = zip(*sorted(zip(abunds, species)))
        VMR_species=species[-n:][::-1] # get n largest
        legend_ncol = int(math.ceil(len(VMR_species)/2))
    elif VMR_species=='all':
        suffix='_all'
        VMR_species = retr_obj.species_names
        legend_ncol = int(math.ceil(len(VMR_species)/4))
    else:
        suffix = '_few'
        legend_ncol = int(math.ceil(len(VMR_species)/4))

    def log_spaced_values(center, num_values=4, log_range=1.0):
        log_center = np.log10(center)
        exponents = np.linspace(log_center - log_range / 2, log_center + log_range / 2, num_values)
        return 10 ** exponents

    def plot_VMRs(retr_obj,ax,ax2):
        
        alpha=1
        if retr_obj.chemistry=='freechem' and 'retr_obj2' not in kwargs:
            linestyle='dashed'
            if retr_obj.target.name in ['Sorg1X','Sorg20X']:
                chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=alpha,label='Retrieved'))
            else:
                chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=alpha,label='Free'))
        elif retr_obj.chemistry=='freechem' and 'retr_obj2' in kwargs:  
            linestyle='dashed'
            chemleg.append(Line2D([0], [0], marker='o',color='k',markerfacecolor='k',linewidth=2,alpha=alpha,label='Free'))
        elif retr_obj.chemistry in ['equchem','flexequ']:
            linestyle='solid'
            alpha=0.8
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=alpha,label='Equ'))
        elif retr_obj.chemistry in ['varchem']:
            linestyle='dashed'
            alpha=0.8
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=alpha,label='Var'))
        elif retr_obj.chemistry=='quequchem':
            linestyle='dotted'
            alpha=0.3
            chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,alpha=alpha,label='Quench'))

        contribution_plot=retr_obj.summed_emcont/np.max(retr_obj.summed_emcont)*(xmax-xmin)+xmin
        global vmr_emcont
        vmr_emcont, = ax2.plot(contribution_plot,pressure,lw=1,alpha=0.5,color=retr_obj.color,linestyle=linestyle)
        ax2.set_xlim(np.min(contribution_plot),np.max(contribution_plot))
        ax2.set_ylim(np.min(pressure),np.max(pressure))
        contr_max=pressure[np.where(retr_obj.summed_emcont==np.max(retr_obj.summed_emcont))[0]]
        ax2.set_yscale('log')
        offset = log_spaced_values(contr_max) # slight vertical offset for overlapping species
        off_i=0

        for species in VMR_species:
            color=retr_obj.species_info.loc[species,'color']
            label=retr_obj.species_info.loc[species,'mathtext_name']
            if retr_obj.chemistry=='freechem' and retr_obj.partialP==False:
                label=label if legend_labels==0 else '_nolegend_' 
                VMR=10**retr_obj.params_dict[f'log_{species}']
                sm3,sm2,sm1,median,sp1,sp2,sp3 = 10**np.array(np.percentile(retr_obj.posterior[f'log_{species}'][0],[0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1))
                if 'retr_obj2' not in kwargs or retr_obj.target.name not in ['Sorg1X','Sorg20X']:
                    ax.plot(np.ones_like(pressure)*VMR,pressure,label=label,linestyle=linestyle,c=color)
                    if suffix!='_all': # only show errors when not showing all species, or will be cluttered
                        ax.fill_betweenx(pressure,sm2,sp2,color=color,alpha=0.1) # 95% confidence interval
                else: # plot only as point to avoid cluttering
                    if (retr_obj.target.name=='2M0355' and species in ['HF','NH3','CH4','HCN']) or (retr_obj.target.name=='2M1425' and species in ['NH3','HF','H2S','HCN']):
                        ax.scatter(VMR,offset[off_i], color=color,s=13)
                        ax.plot([sm2,sp2],[offset[off_i],offset[off_i]], color=color,lw=1.5)
                        off_i+=1
                    else:
                        ax.scatter(VMR,contr_max, color=color,s=13)
                        ax.plot([sm2,sp2],[contr_max,contr_max], color=color,lw=1.5)
            elif (retr_obj.chemistry in ['equchem','quequchem','flexequ','varchem']) or (retr_obj.chemistry=='freechem' and retr_obj.partialP==True):
                label=label if legend_labels==0 else '_nolegend_'
                if comp_equ==False: # compare with actual retrievals
                    sm3,sm2,sm1,median,sp1,sp2,sp3=np.percentile(retr_obj.VMR_dict[species], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=0)
                    ax.plot(median,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                    if retr_obj.chemistry!='quequchem':
                        ax.fill_betweenx(pressure,sm2,sp2,color=color,alpha=0.1) # 95% confidence interval
                    else:
                        ax.fill_betweenx(pressure,sm2,sp2,color=color,alpha=0.05,hatch='x') # 95% confidence interval
                else: # compare with computed equchem based on same params
                    ax.plot(retr_obj.model_object.VMR_dict[species],pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
    
    #ax2 = ax.inset_axes([0,0,1,1]) # [x0, y0, width, height] , for emission contribution
    ax2 = ax.inset_axes([1,0,0.1,1])
    plot_VMRs(retr_obj,ax=ax,ax2=ax2)
    legend_labels=1 if 'retr_obj2' not in kwargs else 0 # only make legend labels once 

    # compare freechem VMRs to equilibrium chemistry with other retrieved params remaining equal
    if comp_equ==True:
        #suffix+='_compequ'
        from retrieval import Retrieval
        from parameters import Parameters
        parameters_equ = retr_obj.params_dict
        parameters_equ.update({'C/O': retr_obj.params_dict['C/O'],
                        'Fe/H': retr_obj.params_dict['C/H']})
        ratios_free,ratios_equ = get_ratios(retr_obj,equ_too=True)
        for r,e in zip(ratios_free,ratios_equ):
            parameters_equ.update({e: retr_obj.params_dict[r]})
        parameters_equ = Parameters({}, parameters_equ)
        parameters_equ.param_priors['log_l']=[-3,0]

        retr_equ = Retrieval(target=retr_obj.target,parameters=parameters_equ, 
                                  species_names=retr_obj.species_names,Nlive=retr_obj.Nlive,
                                  evtol=retr_obj.evtol,chemistry='equchem',
                                  PT_type=retr_obj.PT_type,cloud_mode=retr_obj.cloud_mode)

        retr_equ.model_object=pRT_spectrum(retr_equ,contribution=True)
        retr_equ.model_flux0=retr_equ.model_object.make_spectrum() # to get contr_em
        retr_equ.summed_emcont= retr_equ.model_object.summed_emcont # average over all orders
        plot_VMRs(retr_equ,ax=ax,ax2=ax2)

        # folder created when initializing retrieval object, delete afterwards
        if os.path.exists(retr_equ.output_dir) and not os.listdir(retr_equ.output_dir):  # Check if folder exists and is empty
            os.rmdir(retr_equ.output_dir)  # Remove empty folder

    if 'retr_obj2' in kwargs: # compare two retrs
        suffix='_2'
        retr_obj2=kwargs.get('retr_obj2')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retr_obj2,ax=ax,ax2=ax2)
        legend_labels=1
        comparison_dir=pathlib.Path(f'{retr_obj.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        output_dir=comparison_dir

    if 'retr_obj3' in kwargs: # compare three retrs
        suffix='_3'
        retr_obj3=kwargs.get('retr_obj3')
        plt.gca().set_prop_cycle(None) # reset color cycle
        plot_VMRs(retr_obj3,ax=ax,ax2=ax2)

    if retr_obj.target.name in ['Sorg1X','Sorg20X']:
        tab = PSG_input(retr_obj.target.name).table
        pres = PSG_input(retr_obj.target.name).pressure
        psg_contr=np.genfromtxt(f'LIFE/{retr_obj.target.name}/{retr_obj.target.name}_contr.txt',skip_header=1,delimiter=' ')
        psg_contr=psg_contr[:,1:] # exclude first column (wavelength)
        psg_contr=np.sum(psg_contr,axis=0)[::-1] # sum over all wavelengths, change order
        psg_contr =psg_contr/np.max(psg_contr)*(xmax-xmin)+xmin
        idx_maxcont=np.where(psg_contr == np.max(psg_contr))[0][0]
        input_P_maxcont = pres[idx_maxcont] # pressure at max emission contribution
        #vmr_emcont.remove() # avoid overcrowding plot
        ax2.plot(psg_contr,pres,linestyle='dotted',lw=1.5,alpha=0.5,color='blueviolet')
        ax2.axhline(y=input_P_maxcont, xmin=0, xmax=1,alpha=0.5, color='blueviolet',linestyle='dashdot')
        ax2.axhline(y=10**retr_obj.params_dict['log_P_maxcont'], xmin=0, xmax=1,alpha=0.5, color=retr_obj.color,linestyle='dashdot')
        ax.axhline(y=input_P_maxcont, xmin=0, xmax=1,alpha=0.5, color='blueviolet',linestyle='dashdot')
        ax.axhline(y=10**retr_obj.params_dict['log_P_maxcont'], xmin=0, xmax=1,alpha=0.5, color=retr_obj.color,linestyle='dashdot')

        for species in tab.columns.tolist():
            if species in VMR_species:
                color=retr_obj.species_info.loc[species,'color']
                ax.plot(tab[species],pres,alpha=1,linestyle='solid',c=color)
        chemleg.append(Line2D([0], [0], color='k',linestyle='solid',linewidth=2,alpha=0.7,label='Input'))
        #chemleg.append(Line2D([0], [0], color='blueviolet', linewidth=2,alpha=0.5, linestyle='dotted',label='Input contr'))
        chemleg.append(Line2D([0], [0], color='blueviolet', linewidth=2,alpha=0.5, linestyle='dashdot',label='Input maxcontr'))
        #chemleg.append(Line2D([0], [0], color=retr_obj.color, linewidth=2,alpha=0.5, linestyle='dotted',label='Retr contr'))
        chemleg.append(Line2D([0], [0], color=retr_obj.color, linewidth=2,alpha=0.5, linestyle='dashdot',label='Retr maxcontr'))

    if comp_equ==True or 'retr_obj2' in kwargs or retr_obj.target.name in ['Sorg1X','Sorg20X']:
        leg2=ax.legend(handles=chemleg,fontsize=fs*0.8,loc='upper left')
        ax.add_artist(leg2)

    if addname==True:
        from matplotlib import colors
        fc=colors.to_rgba(retr_obj.color)
        fc = fc[:-1] + (0.5,) # <--- Change the alpha value of facecolor to be 0.7
        ax.annotate(retr_obj.target.name, xy=(0.11,0.08), xycoords='axes fraction',
                    ha='center', va='center', c='k', fontsize=fs*1.1, 
                    bbox={'boxstyle':'round', 'fc':fc, 'ec':'k'})
    
    ax2.axis('off')
    ax2.invert_yaxis()
    ax2.set_facecolor('none')
    ax.set(xlabel='Volume mixing ratio', ylabel='Pressure [bar]',yscale='log',xscale='log',
        ylim=(np.max(pressure),np.min(pressure)),xlim=(xmin,xmax))   
    ax.tick_params(labelsize=fs)
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax.set_xlabel('Volume mixing ratio', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)
    if 'ax' in kwargs and plotlegend==False:
        handles, labels = ax.get_legend_handles_labels()
        line_props=[]
        for handle,label in zip(handles, labels):
            line_props.append((handle.get_color(),label))
        return line_props
    else:
        leg_fs = fs*0.8 if '_all' not in suffix else fs*0.6
        leg=ax.legend(fontsize=leg_fs,ncol=legend_ncol,loc='upper right')
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        for line in leg.get_lines():
            line.set_linestyle('-')
        ax.add_artist(leg)
        if 'ax' not in kwargs:
            fig.tight_layout()
            fig.savefig(f'{output_dir}/{prefix}VMRs{suffix}.pdf')
            plt.close()

def CCF_plot_all(retr_obj,ccf_species,noiserange=100,show_ACF=False,suffix='_all',**kwargs): # plot all CCFs

    RVs=np.arange(-500,500,1) # km/s
    number=len(ccf_species)
    nrows=number//2+number%2
    ncols=2 if len(ccf_species) >1 else 1
    figsize = (5,nrows*1.3)
    if len(ccf_species)==1:
        figname = f'CCF_{ccf_species}' if isinstance(ccf_species, list)==False else f'CCF_{ccf_species[0]}'
        figsize= (4,2.5)
    else:
        figname=f'CCFs{suffix}' if 'retr_obj2' not in kwargs else 'comparison/CCFs_both'

    fig,axes = plt.subplots(nrows,ncols,figsize=figsize,dpi=200,sharex=True)
    for j,species_i in enumerate(ccf_species):
        if len(ccf_species)==1:
            ax=axes
            ccf_species=[ccf_species] if isinstance(ccf_species, list)==False else ccf_species
        else:
            ax=axes[j//2,j%2]
        CCF_norm,ACF_norm,SNR = retr_obj.ccf_acf_dict[species_i]
        if j%2==1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

        ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
        ax.set_xlim(-300,300)
        ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
        ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
        ax.plot(RVs,CCF_norm,color=retr_obj.color,label='CCF')
        if show_ACF:
            ax.plot(RVs,ACF_norm,color=retr_obj.color,label='ACF',linestyle='dashed',alpha=0.5)
            if j==0:
                ax.legend(loc='upper right')
        mathtext_label = retr_obj.species_info.loc[species_i,'mathtext_name']
        if 'retr_obj2' in kwargs: 
            retr_obj2=kwargs.get('retr_obj2')
            CCF_norm2,_,SNR2 = retr_obj2.ccf_acf_dict[species_i]
            ax.plot(RVs,CCF_norm2,color=retr_obj2.color,label='CCF')
            species_label=f'{mathtext_label}'
            lines = [Line2D([0], [0], color=retr_obj.color, linewidth=2,label=retr_obj.target.name),
            Line2D([0], [0], color=retr_obj2.color, linewidth=2,label=retr_obj2.target.name)]
            leg=axes[0,0].legend(handles=lines,fontsize=11,ncol=2,bbox_to_anchor=(1,1.4),loc='upper center')
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_alpha(None)
            leg.get_frame().set_facecolor((0, 0, 0, 0))
            leg.get_frame().set_edgecolor((0, 0, 0, 0))
        else:
            species_label=f'{mathtext_label}\nS/N={np.round(SNR,decimals=1)}'
        ax.text(0.05, 0.9, species_label,transform=ax.transAxes,fontsize=10,verticalalignment='top')

    # in case of odd number, remove last plot
    if len(ccf_species)%2==1 and len(ccf_species)!=1:
        axes[-1,-1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.supxlabel(r'$v_{\rm rad}$ [km/s]')
    fig.supylabel('S/N')
    if len(ccf_species)==1:
        fig.tight_layout() # to avoid ax suplabels to overap with ticklabels
    fig.savefig(f'{retr_obj.output_dir}/{figname}.pdf', bbox_inches='tight')
    plt.close()

def residuals_species(retr_obj,check_species=[],use_equ=True):

    from retrieval import Retrieval
    from parameters import Parameters

    if check_species==[]:
        #check_species = list(retr_obj.species_info.index) # all species, first column
        leave_out= ['H2','He','13CO','C18O','C17O','H2(18)O','H2(17)O','13CH4']
        for species_i in list(retr_obj.species_info.index.values):
            if species_i not in retr_obj.species_names + leave_out:
                check_species.append(species_i)
        print('Checking ', check_species)
    if isinstance(check_species, list)==False:
        check_species=[check_species]

    retr=retr_obj
    residuals=(retr.data_flux-retr.model_flux)

    for species_i in check_species:
        
        # create retrieval object containing only species at equibilrium abundance
        hill_i = retr_obj.species_info.loc[species_i,'Hill_notation']
        equ_table = pathlib.Path(f'{path_tables}/{hill_i}.hdf5')
        parameters_spec = retr_obj.params_dict
        if equ_table.exists() and use_equ:
            suffix=' equ'
            usechem= 'equchem'
            parameters_spec.update({'C/O': retr_obj.params_dict['C/O'],
                            'Fe/H': retr_obj.params_dict['C/H']})
            ratios_free,ratios_equ = get_ratios(retr_obj,equ_too=True)
            for r,e in zip(ratios_free,ratios_equ):
                parameters_spec.update({e: retr_obj.params_dict[r]})
        else:
            suffix = ' logVMR=-5'
            usechem= 'freechem'
            for other_spec_i in retr_obj.species_names:
                parameters_spec.pop(f'log_{other_spec_i}', None)
            parameters_spec[f'log_{species_i}']=-5 # manually set abundance

        parameters_spec = Parameters({}, parameters_spec)
        parameters_spec.param_priors['log_l']=[-3,0]
        retr_spec = Retrieval(target=retr_obj.target,parameters=parameters_spec, 
                                species_names=retr_obj.species_names,Nlive=retr_obj.Nlive,
                                evtol=retr_obj.evtol,chemistry=usechem,
                                PT_type=retr_obj.PT_type,cloud_mode=retr_obj.cloud_mode)
        retr_spec.primary_label=True
        retr_spec.species_names = [species_i]
        retr_spec.species_pRT, retr_spec.species_hill =retr_spec.get_pRT_hill(retr_spec.species_names)
        retr_spec.atmosphere_objects = retr_spec.get_atmosphere_objects(for_species=species_i)
        species_flux=pRT_spectrum(retr_spec).make_spectrum()
        # folder created when initializing retrieval object, delete afterwards
        if os.path.isdir(retr_spec.output_dir) and not os.listdir(retr_spec.output_dir):  # Check if folder exists and is empty
            os.rmdir(retr_spec.output_dir)  # Remove empty folder

        figs=[]
        for part in range(retr_obj.n_parts):

            if np.nansum(residuals[part])==0: # skip empty orders
                continue

            fig,ax=plt.subplots(1,1,figsize=(6,2.5),dpi=200)
            sp_flux = species_flux[part]-np.nanmedian(species_flux[part])

            ax.plot(retr.data_wave[part],residuals[part],lw=0.8,alpha=1,c='k')
            ax.set_xlim(np.nanmin(retr.data_wave[part]),np.nanmax(retr.data_wave[part]))
            ax.set_ylim(np.nanmin([np.nanmin(residuals[part]),np.nanmin(sp_flux)]),
                        np.nanmax([np.nanmax(residuals[part]),np.nanmax(sp_flux)]))
            ax.plot(retr.data_wave[part],np.zeros_like(retr.data_wave[part]),lw=0.8,alpha=0.5,c='k')
            ax.set_ylabel('Residuals')
            ax.set_xlabel('Wavelength [nm]')
            label = f"{retr_obj.species_info.loc[species_i,'mathtext_name']}{suffix}"
            ax.plot(retr.data_wave[part],sp_flux,lw=0.8,c='orange',label=label)
            ax.legend()
            fig.tight_layout()
            figs.append(fig)

        res_dir = pathlib.Path(f'{retr_obj.output_dir}/residuals')
        if use_equ:
            res_dir = pathlib.Path(f'{retr_obj.output_dir}/residuals_equ')
        res_dir.mkdir(parents=True, exist_ok=True)
        with PdfPages(f'{retr_obj.output_dir}/residuals/residuals_{species_i}.pdf') as pdf:
            for fig in figs:
                plt.figure(fig.number)
                pdf.savefig()
                plt.close()

def plot_phi_components(retr_obj): # plot linear function of BD contribution
    wl = retr_obj.data_wave.flatten()
    phi_k = 10**retr_obj.params_dict['log_phi_k']
    phi_d = retr_obj.params_dict['phi_d']
    wl_mid = np.median(retr_obj.data_wave)
    phi = phi_k*(wl-wl_mid)+phi_d
    prim_c='orange'
    sec_c='dodgerblue'
    k = np.round(retr_obj.params_dict['log_phi_k'],decimals=2)
    d = np.round(retr_obj.params_dict['phi_d'],decimals=2)

    slopes = 10**retr_obj.posterior['log_phi_k'][0]
    intercepts = retr_obj.posterior['phi_d'][0]
    y_med,ym1,yp1,ym2,yp2,ym3,yp3 = get_sigma123_lin_func(wl,slopes,intercepts,wl_mid)

    fig,ax=plt.subplots(1,1,figsize=(4,2.5),dpi=200)
    a=0.15
    for (m,p) in zip([ym1,ym2,ym3],[yp1,yp2,yp3]):
        plt.fill_between(wl, m, p, color=sec_c, alpha=a)
        plt.fill_between(wl, np.ones_like(m)-m, np.ones_like(p)-p, color=prim_c, alpha=a)

    plt.plot(wl,phi,color=sec_c,label='B')
    plt.plot(wl,1-phi,color=prim_c,label='A')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Contribution')
    plt.xlim(np.min(wl),np.max(wl))
    plt.legend()
    #plt.text(np.min(wl), np.max(phi)*0.9, f'log $k$ = {k}\n$d$ = {d}')
    ax.text(0.05, 0.5, f'log $k$ = {k}\n$d$ = {d}', transform=ax.transAxes,
            ha='left', va='center')
    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{retr_obj.output_dir}/{prefix}contrast.pdf', bbox_inches='tight')
    plt.close()
    return

def plot_rk(retr_obj): # plot linear function of veiling rk
    wl = retr_obj.data_wave.flatten()
    rk_k = 10**retr_obj.params_dict['log_k_rk']
    rk_d = retr_obj.params_dict['d_rk']
    wl_mid = np.median(retr_obj.data_wave)
    rk = rk_k*(wl-wl_mid)+rk_d
    k = np.round(retr_obj.params_dict['log_k_rk'],decimals=2)
    d = np.round(retr_obj.params_dict['d_rk'],decimals=2)

    slopes = 10**retr_obj.posterior['log_k_rk'][0]
    intercepts = retr_obj.posterior['d_rk'][0]
    y_med,ym1,yp1,ym2,yp2,ym3,yp3 = get_sigma123_lin_func(wl,slopes,intercepts,wl_mid)

    fig,ax=plt.subplots(1,1,figsize=(4,2.5),dpi=200)
    c='yellowgreen'
    a=0.15
    plt.plot(wl,rk,color=c)
    for (m,p) in zip([ym1,ym2,ym3],[yp1,yp2,yp3]):
        plt.fill_between(wl, m, p, color=c, alpha=a)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Veiling factor r$_k$')
    plt.xlim(np.min(wl),np.max(wl))
    #plt.text(np.min(wl), np.max(rk)*0.9, f'log $k$ = {k}\n$d$ = {d}')
    ax.text(0.05, 0.95, f'log $k$ = {k}\n$d$ = {d}', transform=ax.transAxes,
            ha='left', va='top')
    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{retr_obj.output_dir}/{prefix}veiling.pdf', bbox_inches='tight')
    plt.close()
    return

def get_sigma123_lin_func(x,slopes,intercepts,x0=0):
    y_samples = np.array([m * (x-x0) + b for m, b in zip(slopes, intercepts)])
    y_med = np.percentile(y_samples, 50, axis=0)
    ym1 = np.percentile(y_samples, 16, axis=0) # m=minus (1 sigma)
    yp1 = np.percentile(y_samples, 84, axis=0) # p=plus (1 sigma)
    ym2 = np.percentile(y_samples, 2.5, axis=0)
    yp2 = np.percentile(y_samples, 97.5, axis=0)
    ym3 = np.percentile(y_samples, 0.15, axis=0)
    yp3 = np.percentile(y_samples, 99.85, axis=0)
    return y_med,ym1,yp1,ym2,yp2,ym3,yp3

def plot_scaled_abunds(retr_obj): # plot abundances of scaled equchem
    
    fig,ax=plt.subplots(1,1,figsize=(4.5,3),dpi=200)
    spec = []
    scale,scale2 = [],[]
    merr,perr=[],[]
    me1,pe1,me2,pe2,me3,pe3 = [],[],[],[],[],[]
    for species_i in retr_obj.species_names:
        if species_i not in ['13CO','C17O','C18O','H2(18)O']:
            spec.append(retr_obj.species_info.loc[species_i,'mathtext_name'])
            scale2.append(retr_obj.params_dict[f'log_a_{species_i}'])
            m,p = retr_obj.params_dict[f'log_a_{species_i}_err']
            merr.append(abs(m))
            perr.append(p)
            m3,m2,m1,median,p1,p2,p3 = np.percentile(retr_obj.posterior[f'log_a_{species_i}'][0],[0.2,2.3,15.9,50.0,84.1,97.7,99.8])
            scale.append(median)
            me1.append(abs(m1-median))
            pe1.append(abs(p1-median))
            me2.append(abs(m2-median))
            pe2.append(abs(p2-median))
            me3.append(abs(m3-median))
            pe3.append(abs(p3-median))
            #plt.errorbar(str(species_i),a,yerr=[m,p],color='purple')
        
    for i in range(0, len(spec), 2):
        ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.2, edgecolor=None)
    plt.xlim(-0.5,len(spec)+0.5)    

    j=1
    for m,p in zip([me3,me2,me1],[pe3,pe2,pe1]):
        plt.errorbar(spec,scale,yerr=[m,p],color=retr_obj.color,
                    fmt='o',markersize=0,elinewidth=4,alpha=0.1*j)
        j+=1
    #plt.errorbar(spec,scale2,yerr=[merr,perr],color='slateblue',
                    #fmt='o',markersize=4,elinewidth=1)
    plt.xticks(rotation=90) 
    plt.xlabel('Species')
    ax.axhline(0,c='k',linestyle='dashed',alpha=0.3)
    #plt.grid(alpha=0.2)
    plt.ylabel(r'log$_{10}$ scaling wrt. solar')
    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{retr_obj.output_dir}/{prefix}scaled_abunds.pdf', bbox_inches='tight')
    plt.close()
    return

def species_opacities(retr_obj):

    atm = retr_obj.atmosphere_objects
    T = retr_obj.model_object.temperature
    pres = np.log10(retr_obj.pressure)

    wl_cm, opas = atm.get_opa(np.array([T])) 
    wl = wl_cm * 1e4 # um
    extent = [wl.min(), wl.max(), pres.min(), pres.max()]
    #X, Y = np.meshgrid(wl, pres)

    n_rows = int(len(retr_obj.species_names))
    fig,axes=plt.subplots(n_rows,1,figsize=(4,n_rows*1.5),dpi=100,sharex=True,sharey=True)
    fig.supxlabel('Wavelength [um]')
    fig.supylabel('log$_10$ Pressure [bar]')
    axes[-1].set_xlim(np.min(wl),np.max(wl))

    for n,pRT_name in enumerate(opas):
        ax=axes[n]
        name = name_value = retr_obj.species_info.index[retr_obj.species_info["pRT_name"] == pRT_name][0]
        target_color = retr_obj.species_info.loc[name,'color']
        label = retr_obj.species_info.loc[name,'mathtext_name']
        cmap = LinearSegmentedColormap.from_list("white_to_color", [(1, 1, 1), target_color])
        #ax.contourf(X,Y,atm.contr_em,30,cmap=cmap,label=name)
        #log_opas = np.log10(np.clip(opas[pRT_name], 1e-10, None))
        im = ax.imshow(opas[pRT_name],
               aspect='auto', cmap=cmap,
               extent=[wl.min(), wl.max(), pres.min(), pres.max()],
               origin='upper')

        ax.invert_yaxis()
        #ax.legend()
        ax.text(0.05, 0.9, label,transform=ax.transAxes,fontsize=10,verticalalignment='top')
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{retr_obj.output_dir}/{prefix}pressure_opacities.pdf', bbox_inches='tight')
    return

