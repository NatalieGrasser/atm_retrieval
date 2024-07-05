import getpass
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

def plot_spectrum(retrieval_object):
    fig,ax=plt.subplots(7,1,figsize=(9,9),dpi=200)
    for order in range(7):
        for det in range(3):
            ax[order].plot(retrieval_object.data_wave[order,det],retrieval_object.data_flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            lower=retrieval_object.data_flux[order,det]-retrieval_object.data_err[order,det]*retrieval_object.final_params['s2_ij'][order,det]
            upper=retrieval_object.data_flux[order,det]+retrieval_object.data_err[order,det]*retrieval_object.final_params['s2_ij'][order,det]
            ax[order].fill_between(retrieval_object.data_wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax[order].plot(retrieval_object.data_wave[order,det],retrieval_object.final_spectrum[order,det],lw=0.8,alpha=0.8,c='c',label='model')
            #ax[order].yaxis.set_visible(False) # remove ylabels because anyway unitless
            if order==0 and det==0:
                ax[order].legend(ncol=3) # to only have it once
        ax[order].set_xlim(np.nanmin(retrieval_object.data_wave[order]),np.nanmax(retrieval_object.data_wave[order]))
    ax[6].set_xlabel('Wavelength [nm]')
    fig.tight_layout(h_pad=0.1)
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum.pdf')
    plt.close()

def plot_spectrum_w_residuals(retrieval_object):
    retrieval=retrieval_object
    residuals=(retrieval.data_flux-retrieval.final_spectrum)
    fig,ax=plt.subplots(20,1,figsize=(10,13),dpi=200,gridspec_kw={'height_ratios':[2,0.9,0.5]*6+[2,0.9]})
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
        tick_spacing=1
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        if x!=18:
            ax3.set_visible(False) # invisible for spacing
        x+=3
    ax[19].set_xlabel('Wavelength [nm]')
    fig.tight_layout(h_pad=-1.7)
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum.pdf')
    plt.close()

def plot_residuals(retrieval_object):
    fig,ax=plt.subplots(7,1,figsize=(9,9),dpi=200)
    residuals=(retrieval_object.data_flux-retrieval_object.final_spectrum)
    for order in range(7):
        for det in range(3):
            ax[order].plot(retrieval_object.data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c='k',label='residuals')
            ax[order].plot(retrieval_object.data_wave[order,det],np.zeros_like(retrieval_object.data_wave[order,det]),lw=0.8,alpha=0.8,c='c')
            if order==0 and det==0:
                ax[order].legend() # to only have it once
        ax[order].set_xlim(np.nanmin(retrieval_object.data_wave[order]),np.nanmax(retrieval_object.data_wave[order]))
    ax[6].set_xlabel('Wavelength [nm]')
    fig.tight_layout(h_pad=0.1)
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_residuals.pdf')
    plt.close()

def plot_pt(retrieval_object):
    
    if retrieval_object.chemistry=='equchem':
        C_O = retrieval_object.final_object.params['C/O']
        Fe_H = retrieval_object.final_object.params['Fe/H']
    if retrieval_object.chemistry=='freechem':
        C_O = retrieval_object.final_object.CO
        Fe_H = retrieval_object.final_object.FeH   
    fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=100)
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

    # compare with Zhang2022 science verification
    PT_Zhang=np.loadtxt(f'{retrieval_object.target.name}/2M0355_PT_Zhang2021.dat')
    p_zhang=PT_Zhang[:,0]
    t_zhang=PT_Zhang[:,1]
    ax.plot(t_zhang,p_zhang,linestyle='dashdot',c='cornflowerblue',linewidth=2)
    
    # plot PT-profile + errors on retrieved temperatures
    ax.plot(retrieval_object.final_object.temperature,
            retrieval_object.final_object.pressure,color='deepskyblue',lw=2)   
    xmin=np.min(np.min(retrieval_object.final_object.temperature))-100
    xmax=np.max(np.max(retrieval_object.final_object.temperature))+100

    if retrieval_object.PT_type=='PT_knot':
        lowers=[]
        uppers=[]
        medians=[]
        for key in ['T4','T3','T2','T1','T0']: # order T4,T3,T2,T1,T0 like log_P_knots
            median=retrieval_object.final_params[key]
            medians.append(median)
            minus_err,plus_err=retrieval_object.final_params[f'{key}_err']
            lowers.append(minus_err+median)
            uppers.append(median+plus_err)
        lower = CubicSpline(retrieval_object.final_object.log_P_knots,lowers)(np.log10(retrieval_object.pressure))
        upper = CubicSpline(retrieval_object.final_object.log_P_knots,uppers)(np.log10(retrieval_object.pressure))
        ax.fill_betweenx(retrieval_object.pressure,lower,upper,color='deepskyblue',alpha=0.2)
        ax.scatter(medians,10**retrieval_object.final_object.log_P_knots,color='deepskyblue')
        xmin=np.min(np.min(lowers))-100
        xmax=np.max(np.max(uppers))+100

    if False:#retrieval_object.PT_type=='PT_grad':

        lowers=[]
        uppers=[]
        medians=[]
        dlnT_dlnP_knots=retrieval_object.final_object.dlnT_dlnP_knots
        log_P_knots=retrieval_object.final_object.log_P_knots
        for key in dlnT_dlnP_knots:
            median=retrieval_object.final_params[key]
            medians.append(median)
            minus_err,plus_err=retrieval_object.final_params[f'{key}_err']
            lowers.append(minus_err+median)
            uppers.append(median+plus_err)

    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]', yscale='log', 
        ylim=(np.nanmax(retrieval_object.final_object.pressure),
              np.nanmin(retrieval_object.final_object.pressure)),xlim=(xmin,xmax))
    
    summed_contr=np.mean(retrieval_object.final_object.contr_em_orders,axis=0) # average over all orders
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.final_object.pressure,linestyle='dashed',lw=1.5,color='gold')

    # https://github.com/cphyc/matplotlib-label-lines
    labelLines(ax.get_lines(),align=False,fontsize=9,drop_label=True)
    lines = [Line2D([0], [0], marker='o', color='deepskyblue', markerfacecolor='deepskyblue' ,linewidth=2, linestyle='-'),
            mpatches.Patch(color='deepskyblue',alpha=0.2),
            Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot'),
            Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot'),
            Line2D([0], [0], color='gold', linewidth=1.5, linestyle='--')]
    labels = ['This retrieval', '1$\sigma$','Sonora Bobcat \n$T=1400\,$K, log$\,g=4.75$','Zhang+2022','Contribution']
    ax.legend(lines,labels,fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}PT_profile.pdf')
    plt.close()

def cornerplot(retrieval_object,only_abundances=False,only_params=None,not_abundances=False):
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

    fig = plt.figure(figsize=(20,20)) # fix size to avoid memory issues
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
    corner.overplot_lines(fig,medians,color='c',lw=1.3,linestyle='solid') # plot median values of posterior

    if retrieval_object.bestfit_params is not None:
        corner.overplot_lines(fig,np.array([retrieval_object.bestfit_params[i] for i in indices]),color='r',lw=1.3,linestyle='solid')

    if only_abundances==True:
        plot_label='abundances'
    elif only_params is not None:
        plot_label='short'
    elif not_abundances==True:
        plot_label='rest'

    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}cornerplot_{plot_label}.pdf')
    plt.close()

def make_all_plots(retrieval_object,only_abundances=False,only_params=None,split_corner=True):
    #plot_spectrum(retrieval_object)
    #plot_residuals(retrieval_object)
    plot_spectrum_w_residuals(retrieval_object)
    plot_pt(retrieval_object)
    if split_corner: # split corner plot to avoid massive files
        cornerplot(retrieval_object,only_abundances=True)
        cornerplot(retrieval_object,not_abundances=True)
    else: # make cornerplot with all parameters
        cornerplot(retrieval_object,only_abundances=only_abundances,only_params=only_params)