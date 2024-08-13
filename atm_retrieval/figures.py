import getpass
from math import e
import os
if getpass.getuser() == "grasser": # when runnig from LEM
    os.environ['OMP_NUM_THREADS'] = '1' # important for MPI
    import atm_retrieval.cloud_cond as cloud_cond
elif getpass.getuser() == "natalie": # when testing from my laptop
    import cloud_cond as cloud_cond

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
warnings.filterwarnings("ignore", category=UserWarning) 

def plot_spectrum_inset(retrieval_object,inset=True,fs=10,**kwargs):

    wave=retrieval_object.data_wave
    flux=retrieval_object.data_flux
    err=retrieval_object.data_err
    flux_m=retrieval_object.final_spectrum

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(8.5,3),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for order in range(7):
        for det in range(3):
            lower=flux[order,det]-err[order,det]*retrieval_object.final_params['s2_ij'][order,det]
            upper=flux[order,det]+err[order,det]*retrieval_object.final_params['s2_ij'][order,det]
            ax[0].plot(wave[order,det],flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            ax[0].fill_between(wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax[0].plot(wave[order,det],flux_m[order,det],lw=0.8,alpha=0.8,c=retrieval_object.color1,label='model')
            ax[1].plot(wave[order,det],flux[order,det]-flux_m[order,det],lw=0.8,c=retrieval_object.color2,label='residuals')
            if order==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval_object.color1, linewidth=2,label='Bestfit'),
                        Line2D([0], [0], color=retrieval_object.color2, linewidth=2,label='Residuals')]
                ax[0].legend(handles=lines,fontsize=fs,ncol=2) # to only have it once
        ax[1].plot(wave[order].flatten(),np.zeros_like(wave[order].flatten()),lw=0.8,alpha=0.5,c='k')
    ax[0].set_ylabel('Normalized Flux',fontsize=fs)
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
            lower=flux[ord,det]-err[ord,det]*retrieval_object.final_params['s2_ij'][ord,det]
            upper=flux[ord,det]+err[ord,det]*retrieval_object.final_params['s2_ij'][ord,det]
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
            axins2.plot(wave[ord,det],flux[ord,det]-flux_m[ord,det],lw=0.8,c=retrieval_object.color2)
            axins2.plot(wave[ord,det],np.zeros_like(wave[ord,det]),lw=0.8,alpha=0.5,c='k')
        axins2.set_xlim(x1, x2)
        axins2.set_xlabel('Wavelength [nm]',fontsize=fs)
        tick_spacing=1
        axins2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        axins2.tick_params(labelsize=fs)
    else:
        ax[1].set_xlabel('Wavelength [nm]',fontsize=fs) # if no inset

    plt.subplots_adjust(wspace=0, hspace=0)
    if 'ax' not in kwargs:
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum_inset.pdf',
                    bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retrieval_object):
    retrieval=retrieval_object
    residuals=(retrieval.data_flux-retrieval.final_spectrum)
    fig,ax=plt.subplots(20,1,figsize=(10,13),dpi=200,gridspec_kw={'height_ratios':[2,0.9,0.57]*6+[2,0.9]})
    x=0
    for order in range(7): 
        ax1=ax[x]
        ax2=ax[x+1]
        if x!=18: # last ax cannot be spacer, or xlabel also invisible
            ax3=ax[x+2] #for spacing
        for det in range(3):
            ax1.plot(retrieval.data_wave[order,det],retrieval.data_flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            lower=retrieval.data_flux[order,det]-retrieval.data_err[order,det]*retrieval.final_params['s2_ij'][order,det]
            upper=retrieval.data_flux[order,det]+retrieval.data_err[order,det]*retrieval.final_params['s2_ij'][order,det]
            ax1.fill_between(retrieval.data_wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax1.plot(retrieval.data_wave[order,det],retrieval.final_spectrum[order,det],lw=0.8,alpha=0.8,c=retrieval_object.color1,label='model')
            ax1.set_xlim(np.nanmin(retrieval.data_wave[order]),np.nanmax(retrieval.data_wave[order]))
            ax2.plot(retrieval.data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c=retrieval_object.color2,label='residuals')
            ax2.plot(retrieval.data_wave[order,det],np.zeros_like(retrieval.data_wave[order,det]),lw=0.8,alpha=0.5,c='k')
            ax2.set_xlim(np.nanmin(retrieval.data_wave[order]),np.nanmax(retrieval.data_wave[order]))
            if x==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval.color1, linewidth=2,label='Bestfit'),
                        Line2D([0], [0], color=retrieval.color2, linewidth=2,label='Residuals')]
                ax1.legend(handles=lines,fontsize=12,ncol=4,bbox_to_anchor=(0.5,1.4),loc='upper center')
        min1=np.nanmin(np.array([retrieval.data_flux[order]-retrieval.data_err[order],retrieval.final_spectrum[order]]))
        max1=np.nanmax(np.array([retrieval.data_flux[order]+retrieval.data_err[order],retrieval.final_spectrum[order]]))
        ax1.set_ylim(min1,max1)
        ax2.set_ylim(np.nanmin(residuals[order]),np.nanmax(residuals[order]))
        ax1.tick_params(labelbottom=False)  # don't put tick labels at bottom
        ax1.tick_params(axis="both")
        ax2.tick_params(axis="both")
        ax1.set_ylabel('Normalized Flux')
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
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum.pdf')
    plt.close()

def plot_pt(retrieval_object,fs=12,**kwargs):

    if retrieval_object.chemistry=='equchem':
        C_O = retrieval_object.final_object.params['C/O']
        Fe_H = retrieval_object.final_object.params['Fe/H']
    if retrieval_object.chemistry=='freechem':
        C_O = retrieval_object.final_object.CO
        Fe_H = retrieval_object.final_object.FeH   

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)
    cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    #cs_colors=['hotpink','fuchsia','crimson','plum']
    #cs_colors=['forestgreen','limegreen','yellowgreen','tab:olive']
    #cs_colors=['mediumseagreen','mediumaquamarine','lightseagreen','limegreen']
    cs_colors=['gold','goldenrod','peru','sandybrown']

    # if pt profile and condensation curve don't intersect, clouds have no effect
    for i,cs in enumerate(cloud_species):
        cs_key = cs[:-3]
        if cs_key == 'KCl':
            cs_key = cs_key.upper()
        P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
        pi=np.where((P_cloud>min(retrieval_object.final_object.pressure))&(P_cloud<max(retrieval_object.final_object.pressure)))[0]
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
            ax.plot(retr_obj.final_object.temperature,
                retr_obj.final_object.pressure,color=retr_obj.color1,lw=2) 
            medians=[]
            errs=[]
            log_P_knots=retr_obj.final_object.log_P_knots
            for key in ['T4','T3','T2','T1','T0']: # order T4,T3,T2,T1,T0 like log_P_knots
                medians.append(retr_obj.final_params[key])
                errs.append(retr_obj.final_params[f'{key}_err'])
            errs=np.array(errs)
            for x in [1,2,3]: # plot 1-3 sigma errors
                lower = CubicSpline(log_P_knots,medians+x*errs[:,0])(np.log10(retr_obj.pressure))
                upper = CubicSpline(log_P_knots,medians+x*errs[:,1])(np.log10(retr_obj.pressure))
                ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color1,alpha=0.15)
            ax.scatter(medians,10**retr_obj.final_object.log_P_knots,color=retr_obj.color1)
            xmin=np.min(lower)-100
            xmax=np.max(upper)+100
            lines.append(Line2D([0],[0],marker='o',color=retrieval_object.color1,markerfacecolor=retrieval_object.color1,
                    linewidth=2,linestyle='-',label=olabel))

        if retr_obj.PT_type=='PTgrad':
            dlnT_dlnP_knots=[]
            derr=[]
            for i in range(5):
                key=f'dlnT_dlnP_{i}'
                dlnT_dlnP_knots.append(retr_obj.final_params[key]) # gradient median values
                derr.append(retr_obj.final_params[f'{key}_err']) # -/+ errors
            derr=np.array(derr) # gradient errors
            T0=retr_obj.final_params['T0']
            err=retr_obj.final_params['T0_err']
            if False:
                for x in range(4): # plot 1-3 sigma errors
                    if x==0:
                        temperature=retr_obj.final_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots,T_base=T0)
                        ax.plot(temperature,retr_obj.final_object.pressure,color=retr_obj.color1,lw=2) 
                    else:   
                        lower=retr_obj.final_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots+x*derr[:,0],
                                                                    T_base=T0+x*err[0])
                        upper=retr_obj.final_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots+x*derr[:,1],
                                                                    T_base=T0+x*err[1])   
                        ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color1,alpha=0.15)
            temperature=retr_obj.final_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots,T_base=T0)
            ax.plot(temperature,retr_obj.final_object.pressure,color=retr_obj.color1,lw=2) 
            # get 1-2-3 sigma of temp_dist, has shape (samples, n_atm_layers)
            quantiles = np.array([np.percentile(retr_obj.temp_dist[:,i], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1) for i in range(retr_obj.temp_dist.shape[1])])
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,0],quantiles[:,-1],color=retr_obj.color1,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,1],quantiles[:,-2],color=retr_obj.color1,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,2],quantiles[:,-3],color=retr_obj.color1,alpha=0.15)
            #for x in [1,2,3]:
                # temp_dist has shape (samples, n_atm_layers)
                #medians,minus_err,plus_err=retr_obj.get_quantiles(retr_obj.temp_dist)
                #lower=medians+x*minus_err
                #upper=medians+x*plus_err
                #ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color1,alpha=0.15)
            #xmin=np.min((lower,upper))-100
            #xmax=np.max((lower,upper))+100
            xmin=np.min((quantiles[:,0],quantiles[:,-1]))-100
            xmax=np.max((quantiles[:,0],quantiles[:,-1]))+100
            lines.append(Line2D([0], [0], color=retr_obj.color1,
                                linewidth=2,linestyle='-',label=object_label))
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
        summed_contr2=np.nanmean(retrieval_object2.final_object.contr_em_orders,axis=0) # average over all orders
        contribution_plot2=summed_contr2/np.max(summed_contr2)*(xmax-xmin)+xmin
        ax.plot(contribution_plot2,retrieval_object2.final_object.pressure,linestyle='dashed',
                lw=1.5,alpha=0.8,color=retrieval_object2.color3)
        lines.append(Line2D([0], [0], color=retrieval_object2.color3, alpha=0.8,linewidth=1.5, 
                            linestyle='--',label=f'{retrieval_object2.target.name} contribution'))
        if retrieval_object2.target.name=='2M0355':
            lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))

    summed_contr=np.nanmean(retrieval_object.final_object.contr_em_orders,axis=0) # average over all orders
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.final_object.pressure,linestyle='dashed',
            lw=1.5,alpha=0.8,color=retrieval_object.color3)
    lines.append(Line2D([0], [0], color=retrieval_object.color3, alpha=0.8,
                        linewidth=1.5, linestyle='--',label=contr_label))
    lines = lines[:1]+[lines[-1]]+lines[1:-1] # move to second position in legend instead of last

    lines.append(Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=1400\,$K, log$\,g=4.75$'))
    
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retrieval_object.final_object.pressure),
        np.nanmin(retrieval_object.final_object.pressure)),xlim=(xmin,xmax))
    
    ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}PT_profile.pdf')
        plt.close()

def cornerplot(retrieval_object,getfig=False,figsize=(20,20),fs=12,
            only_abundances=False,only_params=None,not_abundances=False):
    
    plot_posterior=retrieval_object.posterior # posterior that we plot here, might get clipped
    medians,_,_=retrieval_object.get_quantiles(retrieval_object.posterior)
    labels=list(retrieval_object.parameters.param_mathtext.values())
    indices=np.linspace(0,len(retrieval_object.parameters.params)-1,len(retrieval_object.parameters.params),dtype=int)
    plot_label='all'

    if only_abundances==True: # plot only abundances
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
                        fig=fig)
    
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
    if retrieval_object.target.name=='test':
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

    if only_abundances==True:
        plot_label='abundances'
    elif only_params is not None:
        plot_label='short'
    elif not_abundances==True:
        plot_label='rest'
        
    if getfig==False:
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}cornerplot_{plot_label}.pdf',
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
    ratios_cornerplot(retrieval_object)
    if retrieval_object.chemistry=='freechem':
        if split_corner: # split corner plot to avoid massive files
            cornerplot(retrieval_object,only_abundances=True)
            cornerplot(retrieval_object,not_abundances=True)
        else: # make cornerplot with all parameters, could be huge
            cornerplot(retrieval_object,only_abundances=only_abundances,only_params=only_params)
    elif retrieval_object.chemistry=='equchem':
        cornerplot(retrieval_object,only_params=only_params)
    

def summary_plot(retrieval_object):

    fs=13
    if retrieval_object.chemistry=='equchem':
        only_params=['rv','vsini','log_g','T0','C/O','Fe/H',
                 'log_C13_12_ratio','log_O18_16_ratio','log_O17_16_ratio']
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
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}summary.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()

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
    molecule_name=retrieval_object.parameters.param_mathtext[f'log_{molecule}'][4:] # remove log_
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

def compare_two_retrievals(retrieval_object1,retrieval_object2,fs=12): # compare cornerplot+PT of two retrievals

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

    figsize=15
    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues
    fig = corner.corner(posterior1, 
                    labels=labels, 
                    title_kwargs={'fontsize': fs},
                    label_kwargs={'fontsize': fs*0.8},
                    color=retrieval_object1.color1,
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
                    fig=fig)
    
    titles = [axi.title.get_text() for axi in fig.axes]
    
    corner.corner(posterior2, 
                    labels=labels, 
                    title_kwargs={'fontsize': fs},
                    label_kwargs={'fontsize': fs*0.8},
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
                    fig=fig)
    
    titles2 = [axi.title.get_text() for axi in fig.axes]

    if False:
        for i,title in enumerate(titles):
            if len(title) > 30: # change 30 to 1 if you want all titles to be split
                title_split = title.split('=')
                title_split2 = titles2[i].split('=')[1] # get only part with values
                titles[i] = title_split[0]+'\n'+title_split[1]+'\n'+title_split2
            fig.axes[i].title.set_text(titles[i])

    for i, axi in enumerate(fig.axes):
        fig.axes[i].title.set_visible(False) # remove original titles
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)
        
    for run,titles_list,color in zip([0,1],[titles,titles2],[retrieval_object1.color1,retrieval_object2.color1]):
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

    plt.subplots_adjust(wspace=0, hspace=0)

    l, b, w, h = [0.58,0.61,0.39,0.39] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retrieval_object1,retrieval_object2=retrieval_object2,ax=ax_PT)

    comparison_dir=pathlib.Path(f'{retrieval_object1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    filename=f'cornerplot_comparison.pdf'
    fig.savefig(f'{comparison_dir}/{filename}',bbox_inches="tight",dpi=200)
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

        filename=f'CCF_{molecule}_comparison.pdf'
        fig.savefig(f'{comparison_dir}/{filename}',bbox_inches="tight",dpi=200)
        plt.close()

def ratios_cornerplot(retrieval_object,fs=10,**kwargs):
    
    labels=[r'log $^{12}$CO/$^{13}$CO',r'log $^{12}$CO/C$^{17}$O',r'log $^{12}$CO/C$^{18}$O',
            r'log H$_2^{16}$O/H$_2^{18}$O','C/O','C/H']
    
    plot_posterior=np.hstack([retrieval_object.ratios_posterior,retrieval_object.CO_CH_dist])
    fig = plt.figure(figsize=(8,8)) # fix size to avoid memory issues
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
                        fig=fig)
    titles = [axi.title.get_text() for axi in fig.axes]
    
    if 'retrieval_object2' in kwargs: # compare two retrievals
        retrieval_object2=kwargs.get('retrieval_object2')
        plot_posterior2=np.hstack([retrieval_object2.ratios_posterior,retrieval_object2.CO_CH_dist])
        corner.corner(plot_posterior2, 
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
                        fig=fig)
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
        filename=f'{comparison_dir}/ratios_comparison.pdf'

    else:
        for i, title in enumerate(titles):
            if len(title) > 1: # change 30 to 1 if you want all titles to be split
                title_split = title.split('=')
                titles[i] = title_split[0] + '\n ' + title_split[1]
            fig.axes[i].title.set_text(titles[i])
        filename=f'{retrieval_object.output_dir}/{retrieval_object.callback_label}ratios.pdf'

    for i, axi in enumerate(fig.axes):
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)

    plt.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(filename,bbox_inches="tight",dpi=200)
    plt.close()

def VMR_plot(retrieval_object,fs=10,**kwargs):

    read_species_info=retrieval_object.final_object.read_species_info

    # convert mass fractions to VMR
    mass_ratio_13CO_12CO = read_species_info('13CO','mass')/read_species_info('12CO','mass')
    mass_ratio_C18O_C16O = read_species_info('C18O','mass')/read_species_info('12CO','mass')
    mass_ratio_C17O_C16O = read_species_info('C17O','mass')/read_species_info('12CO','mass')
    mass_ratio_H218O_H2O = read_species_info('H2(18)O','mass')/read_species_info('H2O','mass')
    
    fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)

    pressure=retrieval_object.final_object.pressure

    if retrieval_object.chemistry=='equchem':
        mass_fractions=retrieval_object.final_object.mass_fractions

        for species in ['H2','He','H2O','CO','CH4','HCN','NH3']:
            ax.plot(mass_fractions[species], pressure, label = species)

    ax.set(xlabel='VMR', ylabel='Pressure [bar]',yscale='log',xscale='log',
        ylim=(np.max(pressure),np.min(pressure)),xlim=(1e-10,1e-1))
    
    #ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    fig.savefig(f'{retrieval_object.output_dir}/VMRs.pdf')
    plt.close()

    