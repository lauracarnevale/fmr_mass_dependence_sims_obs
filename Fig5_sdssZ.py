#Figure 5

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm

import helpers

data_dirs = './Data/'

sims = ['C17O3N2', 'M13O3N2', 'PP04O3N2', 'C17N2', 'M13N2', 'PP04N2', 'M91', 'KK04', 'D16']
thresholds = [-0.5] #[-0.1, -0.5, -1, -np.inf]

def get_data(sim, thresh,  n_bootstrap=1000, plot=False):
    
    m_star_min = 8.0
    m_star_max = 12.0

    currentDir = data_dirs + 'SDSS0/'

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

        #########

    med_sm, med_Z = helpers.get_medians(star_mass, Zgas)
    MZR = interp1d(med_sm, med_Z, fill_value='extrapolate')

    MZR_val = MZR(star_mass)

    MZR_offset = Zgas - MZR_val

    bin_width = 0.25
    mass_bins = np.arange(8.0,12.0,bin_width)

    slopes      = np.zeros(len(mass_bins))
    slopes_mid  = np.zeros(len(mass_bins))
    slopes_low  = np.zeros(len(mass_bins))
    slopes_high = np.zeros(len(mass_bins))

    nums = np.zeros(len(mass_bins))

    for index, mass in enumerate(mass_bins):
        mass_bin_of_interest = mass + bin_width/2

        mask = (star_mass > mass_bin_of_interest - bin_width/2) & (star_mass < mass_bin_of_interest + bin_width/2)

        x, y = sSFR[mask], MZR_offset[mask]

        nums[index] = len(x)

        if len(x) > 50:

            bs_slopes = np.ones(n_bootstrap) * np.nan
            for _ in range(n_bootstrap):
                rand_sample = np.random.randint(0,len(x),len(x))

                tmp_x = x[rand_sample]
                tmp_y = y[rand_sample]
                slope, intercept = np.polyfit(tmp_x, tmp_y, 1)

                bs_slopes[_] = slope

            slopes_mid [index] = np.nanmedian(bs_slopes)
            slopes_low [index] = np.nanpercentile(bs_slopes, 16)
            slopes_high[index] = np.nanpercentile(bs_slopes, 84)
            
        else:
            slopes_mid [index] = np.nan
            slopes_low [index] = np.nan
            slopes_high[index] = np.nan

        
    return mass_bins + bin_width/2, slopes_mid, slopes_low, slopes_high, nums        
        
if __name__ == "__main__":
    
    fig = plt.figure(figsize=(8,6))
    ax  = plt.gca()
    
    colors  = ['orange','orangered','mediumvioletred','darkorchid','navy','deepskyblue','lightseagreen','limegreen', 'green']
    markers = ['o','^','s','d','X', 'X', 'd', 's', '^', 'o']
    dm      = [0.05,0.00,-0.05,0.00,0.05,0, -0.05, 0, 0.05, 0]
    dtext   = [0,-0.07,-0.14, 0, -0.07, -0.14, 0,-0.07,-0.14]
    dx      = [0, 0, 0, 0.2, 0.2, 0.2, 0.36, 0.36, 0.36]
    dtx     = [-0.05, 0.25, -0.3, 0.25]
    dty     = [0, 0, -0.5, -0.5]

    for l, thresh in enumerate(thresholds):
        print(thresh)
        for index, sim in enumerate(sims):
            print(sim)
            mass, slopes_mid, slopes_low, slopes_high, nums = get_data(sim, thresh)
                
            yerr_low  = slopes_mid - slopes_low
            yerr_high = slopes_high - slopes_mid
        
            ax.errorbar(mass+dm[index], slopes_mid, yerr=[yerr_low, yerr_high], 
                    color=colors[index], ms = 4, marker=markers[index], linestyle='none')
        
            text = sim.upper()

            ax.text(0.05+dx[index],0.925+dtext[index],r'${\rm %s}$' %text, fontsize = 17, transform=ax.transAxes, color=colors[index])
            print(nums)

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


        ax.set_xlim(8.0, 11.5)
        ax.set_xticks([8,9,10,11])


        ax.axhline(0.0, color='gray', ls='--', alpha=0.5)
    
        ax.set_ylim(-0.35 , 0.35)

    ax.set_ylabel(r'$\eta_{SFR}$')
    ax.set_xlabel(r'$\log (M_*~[M_\odot])$')

    plt.savefig('./DataGraphs/SDSSmetal.png' , bbox_inches='tight')
