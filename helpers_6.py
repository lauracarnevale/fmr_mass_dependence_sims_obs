# helpers file for Figure 6

import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 2.25*1.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*1.25
mpl.rcParams['ytick.major.width'] = 1.5*1.25
mpl.rcParams['xtick.minor.width'] = 1.0*1.25
mpl.rcParams['ytick.minor.width'] = 1.0*1.25
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

def line(x, a, b):
    return a*x + b

def switch_sim(WHICH_SIM):
    if (WHICH_SIM.upper() == "TNG"):
        # TNG
        run       = 'L75n1820TNG'
        base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
        out_dir   = base 
        snapshots = [99, 91, 78, 67, 59, 50] 
        snap2z = {
            99:'z=0',
            91:'z=0.1',
            78:'z=0.3',
            67:'z=0.5',
            59:'z=0.7',
            50:'z=1',
            33:'z=2',
            25:'z=3',
            21:'z=4',
            17:'z=5',
            13:'z=6',
            11:'z=7',
            8 :'z=8',
            6 :'z=9',
            4 :'z=10',
        }
    elif (WHICH_SIM.upper() == "ORIGINAL"):
        # Illustris
        run       = 'L75n1820FP'
        base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
        out_dir   = base
        snapshots = [135, 127, 113, 103, 95, 85] 
        snap2z = {
            135:'z=0',
            127:'z=0.1',
            113:'z=0.3',
            103:'z=0.5',
            95:'z=0.7',
            85:'z=1',
            86 :'z=1',
            68 :'z=2',
            60 :'z=3',
            54 :'z=4',
            49 :'z=5',
            45 :'z=6',
            41 :'z=7',
            38 :'z=8',
            35 :'z=9',
            32 :'z=10',
        }
    elif (WHICH_SIM.upper() == "EAGLE"):
        snapshots = [28, 27, 25, 23, 21, 19] 
        snap2z = {
            28:'z=0',
            27:'z=0.1',
            25:'z=0.3',
            23:'z=0.5',
            21:'z=0.7',
            19:'z=1',
            15:'z=2',
            12:'z=3',
            10:'z=4',
             8:'z=5',
             6:'z=6',
             5:'z=7',
             4:'z=8',
             3:'z=9',
             2:'z=10'
        }
    elif (WHICH_SIM.upper() == "SIMBA"):
        snapshots = [151, 145, 134, 125, 116, 105] 
        snap2z = {
            151:'z=0',
            145:'z=0.1',
            134:'z=0.3',
            125:'z=0.5',
            116:'z=0.7',
            105:'z=1',
             79:'z=2',
             62:'z=3',
             51:'z=4',
             42:'z=5',
             36:'z=6',
             30:'z=7',
             26:'z=8'
        }

    return snapshots , snap2z


def sfmscut(m0, sfr0, THRESHOLD=-5.00E-01,m_star_min=8.0,m_star_max=12.0):
    '''Compute specific star formation main sequence
    
    Adapted from Z.S.Hemler+(2021)
    
    Inputs:
    - m0 (ndarray): mass array
    - sfr0 (ndarray): SFR array
    - THRESHOLD (float): value below which galaxies omitted
    - m_star_min (float): minimum stellar mass
    - m_star_max (float): maximum stellar mass
    
    Returns:
    - (ndarray): boolean array of systems that meet criteria
    '''
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.0200E+01
    mstp    = 5.0000E-02
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00
    
    nonans = (~(np.isnan(mcs)) &
              ~(np.isnan(rdgs)) &
              ~(np.isnan(rdgs)))
        
    parms, cov = curve_fit(line, mcs[nonans], rdgs[nonans], sigma = rdgstds[nonans])
    mmin    = mbrk
    mmax    = m_star_max
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
        
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool

def get_medians(x,y,width=0.05,min_samp=15):
    '''Get the medians metallicity within fixed mass bins
    
    Inputs:
    - x (ndarray): masses
    - y (ndarray): metallicities
    - width (float): mass bin width
    - min_sample (int): minimum number of galaxies in a bin
    
    Returns:
    - (ndarray): median mass bins
    - (ndarray): corresponding metallicity bins
    - (ndarray): corresponding SFR bins
    '''
    start = np.min(x)
    end   = np.max(x)
    
    xs = np.arange(start,end,width)
    median_y = np.zeros( len(xs) )
    
    for index, current in enumerate(xs):
        mask = ((x > (current)) & (x < (current + width)))
        
        if (len(y[mask]) > min_samp):
            median_y[index] = np.median(y[mask])
        else:
            median_y[index] = np.nan
        
    nonans = ~(np.isnan(median_y))
    
    xs = xs[nonans] + width
    median_y = median_y[nonans]

    return xs, median_y
