# Figure 1


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
thresholds = [-np.inf] 

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
                (sfms_idx) & (sSFR1 > -11))

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
        
        for index, sim in enumerate(sims):
            print(sim)
            Zgas, Mstar, sSFR = get_data(sim, thresh)
            
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

            plot = axs[index].hist2d(Mstar, sSFR, bins=75, norm='log',cmap='Greys')

            axs[index].text(0.06+dx[index], 0.07+dy[index], r'{\rm %s}' %text, fontsize = 15, transform=axs[index].transAxes, ha='left')
                 
            axs[2].set_ylabel(r'$\log(\rm sSFR [yr^{-1}])$')
            

            axs[index].set_xlim(8, 12)

            if index == 3:
                axs[3].text(8.05, -13.1, '8', ha='left', fontsize = 20)
                axs[3].text(9.85, -13.1, '10', ha='left', fontsize = 20)
                axs[3].text(11.85, -13.1, '12', ha='left', fontsize = 20)

            axs[index].set_ylim(-12.5, -7.5)

            if index == 4:
                axs[4].text(1.0,-1.3,r'$\log(\rm M_{*}~[{\rm M_{\odot}}])$', ha='center', transform=axs[2].transAxes)

            Hist1, xedges, yedges = np.histogram2d(Mstar,sSFR,weights=Zgas,bins=(75,75))

            bin_centers = 0.5 * (xedges[1:] + xedges[:-1])
            counts, _, binnumber = binned_statistic(Mstar, sSFR, statistic='count', bins=xedges)
            median_ssfr, _, _ = binned_statistic(Mstar, sSFR, statistic='median', bins=xedges)

            mask = counts > 20

            bin_centers = bin_centers[mask]
            median_ssfr = median_ssfr[mask]

            mass_limit_mask = bin_centers <= 10.2
            
            axs[index].plot(bin_centers[mass_limit_mask], median_ssfr[mass_limit_mask], color='red', linewidth=2)

            cut = median_ssfr - 0.5

            axs[index].plot(bin_centers[mass_limit_mask], cut[mass_limit_mask], color='blue', linewidth=2, ls='--')

            popt, pcov = curve_fit(
            line,
            bin_centers[mass_limit_mask],
            median_ssfr[mass_limit_mask]
            )

            # Predict SFMS at higher masses
            high_mass = np.linspace(10.2, max(Mstar), 100)
            sfms_prediction = line(high_mass, *popt)
            
            mass_mask = bin_centers >= 10.2

            axs[index].plot(high_mass, sfms_prediction, color='red', lw=2) #plot red line

            cut1 = sfms_prediction - 0.5

            axs[index].plot(high_mass, cut1, color='blue', lw=2, ls='--')


        fig.delaxes(axs[5])

        
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig('./DataGraphs/cut.png', bbox_inches='tight')

        
