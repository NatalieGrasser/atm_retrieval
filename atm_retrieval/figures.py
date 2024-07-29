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
warnings.filterwarnings("ignore", category=UserWarning) 

def plot_spectrum_inset(retrieval_object,inset=True,**kwargs):

    fontsize=kwargs.pop('fontsize',None)
    if fontsize!=None:
        plt.rcParams.update({'font.size': fontsize})

    wave=retrieval_object.data_wave
    flux=retrieval_object.data_flux
    err=retrieval_object.data_err
    flux_m=retrieval_object.final_spectrum

    ax=kwargs.pop('ax',None) # ax=None if not passed
    if 'ax'==None:
        #print('making figure and ax')
        fig,ax=plt.subplots(2,1,figsize=(8.5,3),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for order in range(7):
        for det in range(3):
            lower=flux[order,det]-err[order,det]*retrieval_object.final_params['s2_ij'][order,det]
            upper=flux[order,det]+err[order,det]*retrieval_object.final_params['s2_ij'][order,det]
            ax[0].plot(wave[order,det],flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            ax[0].fill_between(wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax[0].plot(wave[order,det],flux_m[order,det],lw=0.8,alpha=0.8,c='c',label='model')
            ax[1].plot(wave[order,det],flux[order,det]-flux_m[order,det],lw=0.8,c='slateblue',label='residuals')
            if order==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2),
                        mpatches.Patch(color='k',alpha=0.15),
                        Line2D([0], [0], color='c', linewidth=2),
                        Line2D([0], [0], color='slateblue', linewidth=2)]
                labels = ['data', '1$\sigma$','model','residuals']
                ax[0].legend(lines,labels,fontsize=9,ncol=2) # to only have it once
        ax[1].plot(wave[order].flatten(),np.zeros_like(wave[order].flatten()),lw=0.8,alpha=0.5,c='k')
    ax[0].set_ylabel('Flux')
    ax[0].set_xlim(np.min(wave)-10,np.max(wave)+10)
    ax[1].set_xlim(np.min(wave)-10,np.max(wave)+10)
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))

    if inset==True:
        ord=5 
        axins = ax[0].inset_axes([0,-1.3,1,0.8])
        for det in range(3):
            lower=flux[ord,det]-err[ord,det]*retrieval_object.final_params['s2_ij'][ord,det]
            upper=flux[ord,det]+err[ord,det]*retrieval_object.final_params['s2_ij'][ord,det]
            axins.fill_between(wave[ord,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            axins.plot(wave[ord,det],flux[ord,det],lw=0.8,c='k')
            axins.plot(wave[ord,det],flux_m[ord,det],lw=0.8,c='c',alpha=0.8)
        x1, x2 = np.min(wave[ord]),np.max(wave[ord])
        axins.set_xlim(x1, x2)
        box,lines=ax[0].indicate_inset_zoom(axins,edgecolor="black",alpha=0.2,lw=0.8,zorder=1e3)
        axins.set_ylabel('Flux')

        axins2 = axins.inset_axes([0,-0.3,1,0.3])
        for det in range(3):
            axins2.plot(wave[ord,det],flux[ord,det]-flux_m[ord,det],lw=0.8,c='slateblue')
            axins2.plot(wave[ord,det],np.zeros_like(wave[ord,det]),lw=0.8,alpha=0.5,c='k')
        axins2.set_xlim(x1, x2)
        axins2.set_xlabel('Wavelength [nm]')
        tick_spacing=1
        axins2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    else:
        ax[1].set_xlabel('Wavelength [nm]') # if no inset

    plt.subplots_adjust(wspace=0, hspace=0)
    if ax==None:
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum_inset.pdf',
                    bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retrieval_object):
    retrieval=retrieval_object
    residuals=(retrieval.data_flux-retrieval.final_spectrum)
    fig,ax=plt.subplots(20,1,figsize=(10,13),dpi=200,gridspec_kw={'height_ratios':[2,0.9,0.65]*6+[2,0.9]})
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
            ax1.plot(retrieval.data_wave[order,det],retrieval.final_spectrum[order,det],lw=0.8,alpha=0.8,c='c',label='model')
            ax1.set_xlim(np.nanmin(retrieval.data_wave[order]),np.nanmax(retrieval.data_wave[order]))
            ax2.plot(retrieval.data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c='slateblue',label='residuals')
            ax2.plot(retrieval.data_wave[order,det],np.zeros_like(retrieval.data_wave[order,det]),lw=0.8,alpha=0.5,c='k')
            ax2.set_xlim(np.nanmin(retrieval.data_wave[order]),np.nanmax(retrieval.data_wave[order]))
            if x==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2),
                        mpatches.Patch(color='k',alpha=0.15),
                        Line2D([0], [0], color='c', linewidth=2),
                        Line2D([0], [0], color='slateblue', linewidth=2)]
                labels = ['data', '1$\sigma$','model','residuals']
                ax1.legend(lines,labels,fontsize=9,ncol=4) # to only have it once
        min1=np.nanmin(np.array([retrieval.data_flux[order]-retrieval.data_err[order],retrieval.final_spectrum[order]]))
        max1=np.nanmax(np.array([retrieval.data_flux[order]+retrieval.data_err[order],retrieval.final_spectrum[order]]))
        ax1.set_ylim(min1,max1)
        ax2.set_ylim(np.nanmin(residuals[order]),np.nanmax(residuals[order]))
        ax1.tick_params(labelbottom=False)  # don't put tick labels at bottom
        ax1.tick_params(axis="both")
        ax2.tick_params(axis="both")
        ax1.set_ylabel('Flux')
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

def plot_pt(retrieval_object,**kwargs):

    fontsize=kwargs.pop('fontsize',None)
    if fontsize!=None:
        plt.rcParams.update({'font.size': fontsize})

    if retrieval_object.chemistry=='equchem':
        C_O = retrieval_object.final_object.params['C/O']
        Fe_H = retrieval_object.final_object.params['Fe/H']
    if retrieval_object.chemistry=='freechem':
        C_O = retrieval_object.final_object.CO
        Fe_H = retrieval_object.final_object.FeH   

    ax=kwargs.pop('ax',None) # ax=None if not passed
    if ax==None: # make separate plot
        fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)
    cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    cs_colors=['hotpink','fuchsia','crimson','plum']

    # if pt profile and condensation curve don't intersect, clouds have no effect
    for i,cs in enumerate(cloud_species):
        cs_key = cs[:-3]
        if cs_key == 'KCl':
            cs_key = cs_key.upper()
        P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
        pi=np.where((P_cloud>min(retrieval_object.final_object.pressure))&(P_cloud<max(retrieval_object.final_object.pressure)))[0]
        ax.plot(T_cloud[pi], P_cloud[pi], lw=1.3, label=cloud_labels[i], ls=':',c=cs_colors[i])
    
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
    
    # plot PT-profile + errors on retrieved temperatures
    ax.plot(retrieval_object.final_object.temperature,
            retrieval_object.final_object.pressure,color='deepskyblue',lw=2)   

    if retrieval_object.PT_type=='PTknot':
        medians=[]
        errs=[]
        log_P_knots=retrieval_object.final_object.log_P_knots
        for key in ['T4','T3','T2','T1','T0']: # order T4,T3,T2,T1,T0 like log_P_knots
            medians.append(retrieval_object.final_params[key])
            errs.append(retrieval_object.final_params[f'{key}_err'])
        errs=np.array(errs)
        for x in [1,2,3]: # plot 1-3 sigma errors
            lower = CubicSpline(log_P_knots,medians+x*errs[:,0])(np.log10(retrieval_object.pressure))
            upper = CubicSpline(log_P_knots,medians+x*errs[:,1])(np.log10(retrieval_object.pressure))
            ax.fill_betweenx(retrieval_object.pressure,lower,upper,color='deepskyblue',alpha=0.15)
        ax.scatter(medians,10**retrieval_object.final_object.log_P_knots,color='deepskyblue')
        xmin=np.min(lower)-100
        xmax=np.max(upper)+100

    if retrieval_object.PT_type=='PTgrad':
        dlnT_dlnP_knots=[]
        derr=[]
        for i in range(5):
            key=f'dlnT_dlnP_{i}'
            dlnT_dlnP_knots.append(retrieval_object.final_params[key]) # median values
            derr.append(retrieval_object.final_params[f'{key}_err']) # -/+ errors
        derr=np.array(derr)
        T0=retrieval_object.final_params['T0']
        err=retrieval_object.final_params['T0_err']
        for i,x in enumerate([1,2,3]): # plot 1-3 sigma errors
            lower=retrieval_object.final_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots+x*derr[:,0],
                                                        T_base=T0+x*err[0])
            upper=retrieval_object.final_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots+x*derr[:,1],
                                                        T_base=T0+x*err[1])   
            ax.fill_betweenx(retrieval_object.pressure,lower,upper,color='deepskyblue',alpha=0.15)
        xmin=np.nanmin((lower,upper))-100
        xmax=np.nanmax((lower,upper))+100
    
    summed_contr=np.nanmean(retrieval_object.final_object.contr_em_orders,axis=0) # average over all orders
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.final_object.pressure,linestyle='dashed',lw=1.5,color='gold')

    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]', yscale='log', 
           ylim=(np.nanmax(retrieval_object.final_object.pressure),
           np.nanmin(retrieval_object.final_object.pressure)),xlim=(xmin,xmax))

    # https://github.com/cphyc/matplotlib-label-lines
    labelLines(ax.get_lines(),align=False,fontsize=9,drop_label=True)
    if retrieval_object.PT_type=='PTknot':
        lines=[Line2D([0],[0],marker='o',color='deepskyblue',markerfacecolor='deepskyblue',
                      linewidth=2,linestyle='-',label='This retrieval')]
    elif retrieval_object.PT_type=='PTgrad':
        lines=[Line2D([0], [0], color='deepskyblue',linewidth=2, linestyle='-',label='This retrieval')]
    lines.append(Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Sonora Bobcat \n$T=1400\,$K, log$\,g=4.75$'))
    lines.append(Line2D([0], [0], color='gold', linewidth=1.5, linestyle='--',label='Contribution'))
    if retrieval_object.target.name=='2M0355':
        lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    ax.legend(handles=lines,fontsize=9)

    if ax==None: # save as separate plot
        fig.tight_layout()
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}PT_profile.pdf')
        plt.close()

def cornerplot(retrieval_object,getfig=False,figsize=(20,20),
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
                        title_kwargs={'fontsize': 12},
                        color='slateblue',
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        fig=fig)
    
    # split title to avoid overlap with plots
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 30: # change 30 to 1 if you want all titles to be split
            title_split = title.split('=')
            titles[i] = title_split[0] + '\n ' + title_split[1]
        fig.axes[i].title.set_text(titles[i])

    corner.overplot_lines(fig,medians,color='b',lw=1.3,linestyle='solid') # plot median values of posterior

    #if retrieval_object.bestfit_params is not None:
        #corner.overplot_lines(fig,np.array([retrieval_object.bestfit_params[i] for i in indices]),color='b',lw=1.3,linestyle='solid')

    # overplot true values of test spectrum
    if False: # didn't work so well bc x-axis range so small, some didn't show up
        if retrieval_object.target.name=='test':
            from testspec import test_parameters,test_mathtext
            compare=np.full(labels.shape,None) # =None for non-input values of test spectrum
            for key_i in test_parameters.keys():
                label_i=test_mathtext[key_i]
                value_i=test_parameters[key_i]
                if label_i in labels:
                    i=np.where(labels==label_i)[0][0]
                    compare[i]=value_i
            corner.overplot_lines(fig,np.array(compare),color='r',lw=1.3,linestyle='solid')

    plt.subplots_adjust(wspace=0, hspace=0)

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
    if split_corner: # split corner plot to avoid massive files
        cornerplot(retrieval_object,only_abundances=True)
        cornerplot(retrieval_object,not_abundances=True)
    else: # make cornerplot with all parameters, could be huge
        cornerplot(retrieval_object,only_abundances=only_abundances,only_params=only_params)
    plot_pt(retrieval_object)
    summary_plot(retrieval_object)

def summary_plot(retrieval_object):

    only_params=['rv','vsini','log_g','T0','log_H2O','log_12CO',
                 'log_13CO','log_HF','log_H2(18)O','log_H2S']
    fig, ax = cornerplot(retrieval_object,getfig=True,only_params=only_params,figsize=(16,16))
    l, b, w, h = [0.37,0.84,0.6,0.15] # left, bottom, width, height
    ax_spec = fig.add_axes([l,b,w,h])
    ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
    plot_spectrum_inset(retrieval_object,ax=(ax_spec,ax_res),inset=False)

    l, b, w, h = [0.68,0.47,0.29,0.29] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retrieval_object,ax=ax_PT)
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
    ax1.plot(RVs,CCF_norm,color='mediumslateblue',label='CCF')
    ax1.plot(RVs,ACF_norm,color='mediumslateblue',linestyle='dashed',alpha=0.5,label='ACF')
    ax1.set_ylabel('S/N')
    ax1.legend(loc='upper right')
    molecule_name=retrieval_object.parameters.param_mathtext[f'log_{molecule}'][4:] # remove log_
    molecule_label=f'{molecule_name}\nS/N={np.round(SNR,decimals=1)}'
    ax1.text(0.05, 0.9, molecule_label,transform=ax1.transAxes,fontsize=14,verticalalignment='top')
    ax2.plot(RVs,CCF_norm-ACF_norm,color='mediumslateblue')
    ax2.set_ylabel('CCF-ACF')
    ax2.set_xlabel(r'$v_{\rm rad}$ (km/s)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f'{retrieval_object.output_dir}/CCF_{molecule}.pdf')
    plt.close()