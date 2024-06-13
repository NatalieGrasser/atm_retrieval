import getpass
import os
if getpass.getuser() == "grasser": # when runnig from LEM
    os.environ['OMP_NUM_THREADS'] = '1' # important for MPI
    from mpi4py import MPI 
    comm = MPI.COMM_WORLD # important for MPI
    rank = comm.Get_rank() # important for MPI
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

def plot_spectrum(retrieval_object):
    fig,ax=plt.subplots(7,1,figsize=(9,9),dpi=200)
    for order in range(7):
        for det in range(3):
            ax[order].plot(retrieval_object.data_wave[order,det],retrieval_object.data_flux[order,det],lw=0.8,alpha=0.8,c='k',label='data')
            ax[order].plot(retrieval_object.data_wave[order,det],retrieval_object.final_spectrum[order,det],lw=0.8,alpha=0.8,c='c',label='model')
            #ax[order].yaxis.set_visible(False) # remove ylabels because anyway unitless
            #sigma=1
            #lower=self.data_flux[order,det]-self.data_err[order,det]*sigma
            #upper=self.data_flux[order,det]+self.data_err[order,det]*sigma
            #ax[order].fill_between(self.data_wave[order,det],lower,upper,color='k',alpha=0.2,label=f'{sigma}$\sigma$')
            if order==0 and det==0:
                ax[order].legend(fontsize=8) # to only have it once
        ax[order].set_xlim(np.nanmin(retrieval_object.data_wave[order]),np.nanmax(retrieval_object.data_wave[order]))
    ax[6].set_xlabel('Wavelength [nm]')
    fig.tight_layout(h_pad=0.1)
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum.pdf')
    plt.close()

def plot_pt(retrieval_object):
    
    if retrieval_object.free_chem==False:
        C_O = retrieval_object.final_object.params['C_O']
        Fe_H = retrieval_object.final_object.params['FEH']
    if retrieval_object.free_chem==True:
        C_O = retrieval_object.final_object.CO
        Fe_H = retrieval_object.final_object.FeH   
    fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=100)
    cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    cs_colors=['hotpink','fuchsia','crimson','plum']

    for i,cs in enumerate(cloud_species):
        cs_key = cs[:-3]
        if cs_key == 'KCl':
            cs_key = cs_key.upper()
        P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
        pi=np.where((P_cloud>min(retrieval_object.final_object.pressure))&(P_cloud<max(retrieval_object.final_object.pressure)))[0]
        ax.plot(T_cloud[pi], P_cloud[pi], lw=1.3, label=cloud_labels[i], ls=':',c=cs_colors[i])
    
    # T=1400K, logg=4.65 -> 10**(4.65)/100 =  446 m/s²
    file=np.loadtxt('t1400g562nc_m0.0.dat')
    pres=file[:,1] # bar
    temp=file[:,2] # K
    ax.plot(temp,pres,linestyle='dashdot',c='blueviolet',linewidth=2)

    # plot errors on retrieved temperatures
    temps=['T1','T2','T3','T4']
    T_pm=np.zeros((len(temps),3))
    for i,key in enumerate(temps):
        T_pm[i]=(retrieval_object.params_pm_dict[key]) # median, lower, upper

    medians=T_pm[:,1][::-1]
    lowers=T_pm[:,0][::-1] # reverse order so that T4,T3,T2,T1, like p_samp
    uppers=T_pm[:,2][::-1]
    lower = CubicSpline(retrieval_object.final_object.p_samp,lowers)(np.log10(retrieval_object.pressure))
    upper = CubicSpline(retrieval_object.final_object.p_samp,uppers)(np.log10(retrieval_object.pressure))
    ax.fill_betweenx(retrieval_object.pressure,lower,upper,color='deepskyblue',alpha=0.2)

    # temperature from medians of final posterior
    #temperature = CubicSpline(self.final_object.p_samp,medians)(np.log10(self.pressure))
    #ax.plot(temperature, self.final_object.pressure,color='deepskyblue',lw=2)
    ax.scatter(medians,10**retrieval_object.final_object.p_samp,color='deepskyblue')
    xmin=np.min(np.min(lowers))-100
    xmax=np.max(np.max(uppers))+100
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]', yscale='log', 
        ylim=(np.nanmax(retrieval_object.final_object.pressure),np.nanmin(retrieval_object.final_object.pressure)),
        xlim=(xmin,xmax))
    
    summed_contr=np.mean(retrieval_object.final_object.contr_em_orders,axis=0) # average over all orders
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.final_object.pressure,linestyle='dashed',lw=1.5,color='gold')

    # https://github.com/cphyc/matplotlib-label-lines
    labelLines(ax.get_lines(),align=False,fontsize=9,drop_label=True)
    lines = [Line2D([0], [0], marker='o', color='deepskyblue', markerfacecolor='deepskyblue' ,linewidth=2, linestyle='-'),
            mpatches.Patch(color='deepskyblue',alpha=0.2),
            Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot'),
            Line2D([0], [0], color='gold', linewidth=1.5, linestyle='--')]
    labels = ['This retrieval', '68%','Sonora Bobcat \n$T=1400\,$K, log$\,g=4.75$','Contribution']
    ax.legend(lines,labels,fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}PT_profile.pdf')
    plt.close()

def cornerplot(retrieval_object,only_abundances=False,only_params=None,not_abundances=False):
    plot_posterior=retrieval_object.posterior # posterior that we plot here, might get clipped
    medians,_,_=retrieval_object.get_quantiles(retrieval_object.posterior,save=True)
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

    fig = corner.corner(plot_posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize': 12},
                        color='slateblue',
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.84],
                        show_titles=True)
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
    plot_spectrum(retrieval_object)
    plot_pt(retrieval_object)
    if split_corner: # split corner plot to avoid massive files
        cornerplot(retrieval_object,only_abundances=True)
        cornerplot(retrieval_object,not_abundances=True)
    else:
        cornerplot(retrieval_object,only_abundances=only_abundances,only_params=only_params)