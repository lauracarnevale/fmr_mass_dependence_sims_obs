#Figure 5

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, ListedColormap

from scipy.stats import binned_statistic

import helpers

mpl.rcParams['text.usetex']         = True
mpl.rcParams['font.family']         = 'serif'
mpl.rcParams['font.size']           = 16
mpl.rcParams['axes.linewidth']      = 2
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width']   = 1.5
mpl.rcParams['ytick.major.width']   = 1.5
mpl.rcParams['xtick.minor.width']   = 1.0
mpl.rcParams['ytick.minor.width']   = 1.0
mpl.rcParams['xtick.major.size']    = 7.5
mpl.rcParams['ytick.major.size']    = 7.5
mpl.rcParams['xtick.minor.size']    = 3.5
mpl.rcParams['ytick.minor.size']    = 3.5
mpl.rcParams['xtick.top']           = True
mpl.rcParams['ytick.right']         = True

data_dirs = './Data/'

sims = ['C17O3N2', 'M13O3N2', 'PP04O3N2', 'C17N2', 'M13N2', 'PP04N2', 'M91', 'KK04', 'D16']
thresholds = [-0.5] #[-0.1, -0.5, -1, -np.inf]

def get_data(sim, thresh,  n_bootstrap=1000, plot=False):
    
    m_star_min = 8.0
    m_star_max = 12.0

    currentDir = data_dirs + '/'

    Zgas    = np.load( currentDir + 'SDSSZgas_' + sim  + '.npy' )
    star_mass   = np.load( currentDir + 'SDSSMstar.npy'  )
    SFR         = np.load( currentDir + 'SDSSSFR.npy' )

    mask1 = ((~np.isnan(star_mass)) & (~np.isnan(SFR)) & (~np.isnan(Zgas)))
    star_mass = star_mass[mask1]
    Zgas = Zgas[mask1]
    SFR = SFR[mask1]

    SFR2 = 10**SFR
    star_mass2 = 10**star_mass
    sSFR1 = np.log10(SFR2/star_mass2)

    sfms_idx = helpers.sfmscut(star_mass2, SFR2, thresh,
                    m_star_min=m_star_min, m_star_max=m_star_max)

    desired_mask = ((star_mass2 > 1.00E+01**(m_star_min)) &
                (star_mass2 < 1.00E+01**(m_star_max)) &
                (sfms_idx) & (sSFR1 > -10.5))

    star_mass2 = star_mass2[desired_mask]
    SFR2       = SFR2      [desired_mask]
    Zgas        = Zgas    [desired_mask]
    sSFR = np.log10(SFR2 / star_mass2)

    star_mass = np.log10(star_mass2)
    
    return star_mass, Zgas, sSFR

def get_data_sim(sim, thresh,  n_bootstrap=1000, plot=False):
    
    snapshots = helpers.switch_sim(sim)
    
    m_star_min = 8.0
    if sim == "SIMBA":
        m_star_min = 9.0
       # m_star_max = 11.0
    m_gas_min  = m_star_min
    m_star_max = 12.0
    
    for j, snap in enumerate(snapshots):
        currentDir = data_dirs + sim + '/' + 'snap%s/' %snap
        if sim == 'SIMBA':
            currentDir = './Data/SIMBA0/snap151/'
    
        Zgas      = np.load( currentDir + 'Zgas.npy' )
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
    
        # Nominal threshold = -5.000E-01
        sfms_idx = helpers.sfmscut(star_mass, SFR, thresh,
                               m_star_min=m_star_min, m_star_max=m_star_max)
    
    
        sSFR1 = np.log10(SFR/star_mass)
    
        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                    (star_mass < 1.00E+01**(m_star_max)) &
                    (gas_mass  > 1.00E+01**(m_gas_min))  &
                    (sfms_idx))
    
        gas_mass  = gas_mass [desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       = SFR      [desired_mask]
        Zgas      = Zgas     [desired_mask]
    
        sSFR = np.log10(SFR / star_mass)
    
        star_mass = np.log10(star_mass)
    
        Zgas = np.log10( Zgas * (0.35/0.76) * (1.00/16.00) ) + 12


    return Zgas, star_mass, sSFR
        
if __name__ == "__main__":
    
    fig, axs = plt.subplots(3,3,figsize=(12,10), sharex=True, sharey=True)
    axs = axs.flatten()
    
    colors  = ['orange','orangered','mediumvioletred','darkorchid','navy','deepskyblue','lightseagreen','limegreen', 'green']
    markers = ['o','^','s','d','X', 'X', 'd', 's', '^', 'o']
    dm      = [0.05,0.00,-0.05,0.00,0.05,0, -0.05, 0, 0.05, 0]
    dtext   = [0,-0.07,-0.14, 0, -0.07, -0.14, 0,-0.07,-0.14]
    dx      = [0, 0, 0, 0.2, 0.2, 0.2, 0.36, 0.36, 0.36]
    dtx     = [-0.05, 0.25, -0.3, 0.25]
    dty     = [0, 0, -0.5, -0.5]

    C17x = None
    C17y = None

    for l, thresh in enumerate(thresholds):
        print(thresh)
        for index, sim in enumerate(sims):
            print(sim)
            Mstar, Zgas, sSFR = get_data(sim, thresh)
                
            ax = axs[index]

            CMIN = -10.1
            CMAX = -8.9
            spacing = 6
            color_bins = np.linspace( CMIN,CMAX,spacing )
            newcolors = plt.cm.viridis(np.linspace(0, 1, len(color_bins)))
            newcmp = plt.cm.viridis#ListedColormap(newcolors) 
    
            Hist1, xedges, yedges = np.histogram2d(Mstar,Zgas,weights=sSFR,bins=(75,75))
            Hist2, _     , _      = np.histogram2d(Mstar,Zgas,bins=[xedges,yedges])
    
            Hist1 = np.transpose(Hist1) 
            Hist2 = np.transpose(Hist2)
            hist = Hist1/Hist2
            
            plot = ax.pcolormesh(xedges,yedges,hist,cmap=newcmp,vmin=CMIN,vmax=CMAX, rasterized=True)

            ax.text(0.05,0.90,r'${\rm %s}$' %sim.upper(), fontsize = 17, transform=ax.transAxes, color='k')

            if index == 0:
                ax_cbar = fig.add_axes([0.13, 1.02, 0.6, 0.02])
                cb = plt.colorbar(plot, cax=ax_cbar,ticks=np.linspace(CMIN,CMAX,spacing+1) ,shrink=0.5,orientation='horizontal')
                cb.set_label(r'$\log({\rm sSFR}~[{\rm yr}^{-1}])$')
                cb.ax.xaxis.set_label_position('top')
                cb.ax.tick_params(labelsize = 18)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            bin_centers = 0.5 * (xedges[1:] + xedges[:-1])
            counts, _, binnumber = binned_statistic(Mstar, Zgas, statistic='count', bins=xedges)
            median_zgas, _, _ = binned_statistic(Mstar, Zgas, statistic='median', bins=xedges)

            mask = counts > 20

            bin_centers = bin_centers[mask]
            median_zgas = median_zgas[mask]

            ax.plot(bin_centers, median_zgas, color='red', linewidth=2)

            if index == 0:
                C17x = bin_centers
                C17y = median_zgas

    sims = ['ORIGINAL','TNG','SIMBA','EAGLE']
    colors  = ['red','orange','mediumvioletred','navy','deepskyblue']

    axs[0].plot(C17x,C17y,color='red',label=r'${\rm Diagnostic}$')
    
    for index, sim in enumerate(sims):
        print(sim)
        Zgas, Mstar, sSFR = get_data_sim(sim, thresh)

        Hist1, xedges, yedges = np.histogram2d(Mstar,Zgas,weights=sSFR,bins=(75,75))

        bin_centers = 0.5 * (xedges[1:] + xedges[:-1])
        counts, _, binnumber = binned_statistic(Mstar, Zgas, statistic='count', bins=xedges)
        median_zgas, _, _ = binned_statistic(Mstar, Zgas, statistic='median', bins=xedges)

        mask = counts > 20

        bin_centers = bin_centers[mask]
        median_zgas = median_zgas[mask]

        text = sim.upper()
        if sim == 'ORIGINAL':
            text = 'Illustris'
        print(text)
        for ax in axs:
            ax.plot(bin_centers, median_zgas, color=colors[index+1], linewidth=2, ls='--', label=text)

    handles, labels = axs[0].get_legend_handles_labels()
    leg = axs[2].legend(handles, labels, frameon=False, fontsize=16, bbox_to_anchor=(1,1))
    for iii, text in enumerate(leg.get_texts()):
        text.set_color(colors[iii])
            

    axs[-1].set_xlabel(r'$\log(M_\star~[{\rm M}_\odot])$')
    axs[-2].set_xlabel(r'$\log(M_\star~[{\rm M}_\odot])$')
    axs[-3].set_xlabel(r'$\log(M_\star~[{\rm M}_\odot])$')

    axs[3].set_ylabel(r'$\log({\rm O/H})+12~[{\rm dex}]$')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig('./DataGraphs/SDSS_MZR.pdf' , bbox_inches='tight')
