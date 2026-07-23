#Figure 2


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, ListedColormap
from scipy.optimize import curve_fit
import helpers
from scipy.stats import binned_statistic
data_dirs = './Data/'

sims = ['ORIGINAL','TNG','SIMBA','EAGLE','SDSS']
thresholds = [-0.5] 

def get_data(sim, thresh,  n_bootstrap=1000, plot=False):
    
    if sim == 'SDSS':
        m_star_min = 8.0
        m_star_max = 12.0

        currentDir = data_dirs

        Zgas      = np.load( currentDir + 'SDSSZgas.npy' )
        star_mass = np.load( currentDir + 'SDSSMstar.npy'  )
        SFR       = np.load( currentDir + 'SDSSSFR.npy' )

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
        Zgas      = Zgas     [desired_mask]

        sSFR = np.log10(SFR2 / star_mass2)

        star_mass = np.log10(star_mass2)


        ##########


    else:
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
        

def line(x, a, b):
    return a*x + b

        
if __name__ == "__main__":
    
    fig = plt.figure(figsize=(8,6))
    ax  = plt.gca()
    
    colors  = ['orange','mediumvioletred','navy','deepskyblue','limegreen']
    markers = ['o','^','s','d','X']
    dm      = [0.05,0.00,-0.05,0.00,0.05]
    dtext   = [0.00,-0.05,-0.1,-0.15,-0.2]
   
    dx = [0, 0, 0, 0, 0]
    dy = [0, 0, 0, 0, 0]

    dx1 = [0.01, 0.51, 0.01, 0.51, 0.01]
    dy1 = [0, 0, -0.45, -0.45, -0.89]

    for l, thresh in enumerate(thresholds):
        print(thresh)
        
        fig, axs_FMR = plt.subplots(3, 2, figsize = (8,8), sharex=True, sharey=True)

        axs = axs_FMR.flatten()
        median_lines = []
        for index, sim in enumerate(sims):
            print(sim)
            Zgas, Mstar, sSFR = get_data(sim, thresh)
            
            CMIN = -10.5
            CMAX = -9.3  
            spacing = 6
            color_bins = np.linspace( CMIN,CMAX,spacing )
            newcolors = plt.cm.viridis(np.linspace(0, 1, len(color_bins)))
            newcmp = ListedColormap(newcolors) 
    
            Hist1, xedges, yedges = np.histogram2d(Mstar,Zgas,weights=sSFR,bins=(75,75))
            Hist2, _     , _      = np.histogram2d(Mstar,Zgas,bins=[xedges,yedges])
    
            Hist1 = np.transpose(Hist1) 
            Hist2 = np.transpose(Hist2)
            hist = Hist1/Hist2

            mpl.rcParams['text.usetex']         = True
            mpl.rcParams['font.family']         = 'serif'
            mpl.rcParams['font.size']           = 24
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

            text = sim.upper()
            if sim == "ORIGINAL":
                text = "ILLUSTRIS"

            plot = axs[index].pcolormesh(xedges,yedges,hist,cmap=newcmp,vmin=CMIN,vmax=CMAX)
            
            axs[index].text(0.06+dx[index], 0.07+dy[index], r'{\rm %s}' %text, fontsize = 15, transform=axs[index].transAxes, ha='left')

            if index == 0:
                ax_cbar = fig.add_axes([0.2, 0.93, 0.6, 0.05])
                cb = plt.colorbar(plot, cax=ax_cbar,ticks=np.linspace(CMIN,CMAX,spacing+1) ,shrink=0.5,orientation='horizontal')
                cb.set_label(r'$\log({\rm sSFR}~[{\rm yr}^{-1}])$')
                cb.ax.xaxis.set_label_position('top')
                cb.ax.tick_params(labelsize = 18)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                 
            axs[2].set_ylabel(r'$\log({\rm O/H}) + 12$')


            axs[index].set_ylim(7.2, 9.6)
            axs[index].set_xlim(8, 11.8)


            if index == 4:
                axs[4].text(1.0,-1.3,r'$\log(\rm M_{*}~[{\rm M_{\odot}}])$', ha='center', transform=axs[2].transAxes)

            bin_centers = 0.5 * (xedges[1:] + xedges[:-1])
            counts, _, binnumber = binned_statistic(Mstar, Zgas, statistic='count', bins=xedges)
            median_zgas, _, _ = binned_statistic(Mstar, Zgas, statistic='median', bins=xedges)

            mask = counts > 20

            bin_centers = bin_centers[mask]
            median_zgas = median_zgas[mask]

            axs[index].plot(bin_centers, median_zgas, color='red', linewidth=2)

            median_lines.append((sim, bin_centers, median_zgas))
    
            if sim == 'SDSS':
                np.savez('median_SDSS.npz', bin_centers=bin_centers, median_zgas=median_zgas)

            xstart = 10.6
            ystart = 8.4
            xlen = 1.2
            ylen = 1

            ax_inset = axs[index].inset_axes([0.25+dx1[index], 1.01+dy1[index], 0.15, 0.11], transform=ax.transData)
            
            # Make an array to hold the per-galaxy median values (same length as Zgas)
            Zgas_bin_medians = np.full_like(Zgas, np.nan, dtype=float)

            # Total number of bins (same as length of median_zgas)
            n_bins = len(median_zgas)

            # Loop over valid bins and assign median values to galaxies in that bin
            for bin_idx in range(1, n_bins + 1):  # bin numbers go from 1 to n_bins
                in_bin = binnumber == bin_idx
                Zgas_bin_medians[in_bin] = median_zgas[bin_idx - 1]  # median_zgas is 0-indexed

            # Now subtract the bin-wise median from each galaxy's Zgas
            deltaZ = Zgas - Zgas_bin_medians



            sSFR = np.array(sSFR)
            deltaZ = np.array(deltaZ)
            mask1 = np.isfinite(sSFR) & np.isfinite(deltaZ)

            sSFR1 = sSFR[mask1]
            deltaZ1 = deltaZ[mask1]
            h = ax_inset.hist2d(sSFR1, deltaZ1,norm='log', bins=75, cmap='Greys')

            m, b = np.polyfit(sSFR1, deltaZ1, 1)
            ax_inset.plot(sSFR1, m*sSFR1 + b, color='deeppink', lw=1)

            print(m)

            dxa=[-0.01, 0.49, -0.01, 0.49, -0.01]
            dya=[0.98, 0.98, 0.53, 0.53, 0.095]

            ax_inset.text(0.365+dxa[index], 0.1+dya[index], f"{m:.2f}",transform=ax.transAxes, fontsize=10, color='deeppink')

            print(m)

            ax_inset.spines['bottom'].set_linewidth(1); ax_inset.spines['top'].set_linewidth(1)
            ax_inset.spines['left'].set_linewidth(1)  ; ax_inset.spines['right'].set_linewidth(1)
        
            ax_inset.tick_params(axis='both', which='major', length=3, width=1.5)
            ax_inset.tick_params(axis='x', which='minor', length=2, width=1)
            ax_inset.tick_params(axis='y', which='minor', length=0)

            ax_inset.tick_params(axis='y', labelsize=9)
            ax_inset.tick_params(axis='x', which='both', top=True, labelsize=9)
    
            ax_inset.set_ylim(-0.5, 0.5)
            ax_inset.set_xlim(-11.6,-7.9)

            ax_inset.text(-0.4,0.5,r'${\rm \Delta Z}$', fontsize=10,
                  transform=ax_inset.transAxes, va='center', rotation=90)
            ax_inset.text(0.5,-0.55,r'${\rm log(sSFR [yr^{-1}])}$', fontsize=10,
                  transform=ax_inset.transAxes, ha='center')

            ax_inset.set_xticks([-10, -8])
            ax_inset.set_yticks([-0.5, 0, 0.5])
    

            ax_summary = axs[5]  # Use panel 6

            colors  = ['orangered', 'limegreen', 'dodgerblue', 'orchid', 'black']
            lw      = [1.7,1.7,1.7,1.7,2]
            ls      = ['--','--','--','--','-']
            a       = [0, 0, 0.4, 0.4, 0.4]
            b       = [-0.05, -0.16, 0.06, -0.05, -0.16]
            
        for i, (sim, x, y) in enumerate(median_lines):
            label = sim.upper() if sim != "ORIGINAL" else "ILLUSTRIS"
            ax_summary.plot(x, y, label=label, lw=lw[i], ls=ls[i], color=colors[i])
               
            ax_summary.text(0.25+a[i], 0.26+b[i] ,r'{\rm %s}' %label, color=colors[i], fontsize = 14, transform=axs[5].transAxes, ha='left')



            ax_summary.set_xlim(8, 11.8)
            ax_summary.set_ylim(7.2, 9.6)



        
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig('./DataGraphs/MZR.png', bbox_inches='tight')

        
