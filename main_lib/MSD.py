# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Contains lots of methods to compute msds and variatons thereof for particles on flat and curved surfaces.

@author: Jack Bond, Alex Yeh
"""

import numpy as np
from numpy.random import default_rng
from scipy.optimize import curve_fit

from GeometryHelpers import chord_to_arc


#MULTIPLE TIME ORIGIN MSDS

def _mto(times, max_lag=None, orig_num=None, delta=None):
    """
    Given a set of times and a maximum lagtime, returns a list of indices
    corresponding to the starting points of each mto trajectory, as well as
    the number of points in each trajectory
    There are multiple ways to do this calculation.
    delta: specify the time between time origins
    orig_num: specify a desired amount of origins (takes precedence)
    source: MSD, 6/15/23
    author: Jack Bond
    """
    
    # if no max lag is given, assume only 1 time origin
    if max_lag is None: max_lag=times[-1]
    #lagtime cannot me more than
    assert max_lag <= times[-1], "max lagtime is larger than total runtime"
    #find the maximum trajectory length
    lag_idx = np.argmin(np.abs(times-max_lag))
    
    total_steps=len(times)

    if orig_num is None:
        #default to one full-length time origin
        if delta is None: time_origins = np.array([0])
        step = np.argmin(np.abs(times-delta))
        time_origins = np.arange(0,total_steps-lag_idx,step)
    else:
        time_origins = np.linspace(0,total_steps-lag_idx,orig_num).astype(int)
    
    #this should be impossible, but here's a stopgap anyway
    assert time_origins[-1]+lag_idx < total_steps, "final step will exceed array size. Either reduce max lagtime or use fewer time origins."

    return time_origins, lag_idx


#MAIN MSD ANALYSIS METHODS


def mto_msd(coords, times, max_lag=None, orig_num = None, delta = None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd up to the given max lagtime.
    MSDs are then averaged together over several subsets of length [TO],
    within the full time-ordered list. The number of, or distance between,
    each time origin may be specified. If neither is specified, no mto
    calculation occurs. 
    Returns a ([TO x 1]) array of msds and a ([TO x 1]) array of lagtimes
    source: MSD, 6/15/23
    author: Jack Bond
    """

    #set up multiple time origins
    time_origins, lag_idx = _mto(times,max_lag=max_lag,orig_num=orig_num,delta=delta)

    #to save memory we loop over time origins and repeatedly add them to the final array
    mto_average = np.zeros(lag_idx)
    for og in time_origins:
        msd_comp = (coords[og:(og+lag_idx)]-coords[og])**2
        msd_full = msd_comp.sum(axis=-1)
        ptcl_average = msd_full.mean(axis=-1)
        mto_average += ptcl_average/len(time_origins)

    return mto_average, times[:lag_idx]


def mto_msd_part(coords, times, max_lag=None, orig_num = None, delta = None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max lagtime.
    MSDs are then averaged together over several subsets of length [TO],
    within the full time-ordered list. The number of, or distance between,
    each time origin may be specified. If neither is specified, no mto
    calculation occurs. 
    Returns a ([TO x N]) array of msds and a ([TO x 1]) array of lagtimes
    source: MSD, 6/15/23
    author: Jack Bond
    """

    #set up multiple time origins
    time_origins, lag_idx = _mto(times,max_lag=max_lag,orig_num=orig_num,delta=delta)

    #to save memory we loop over time origins and repeatedly add them to the final array
    mto_average = np.zeros((lag_idx,coords.shape[1]))
    for og in time_origins:
        msd_comp = (coords[og:(og+lag_idx)]-coords[og])**2
        msd_full = msd_comp.sum(axis=-1)
        mto_average += msd_full/len(time_origins)

    return mto_average, times[:lag_idx]


def mto_msd_arc(coords, times, max_lag=None, orig_num = None, delta = None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd in the arclength coordinate up to the given max lagtime.
    MSDs are then averaged together over several subsets of length [TO],
    within the full time-ordered list. The number of, or distance between,
    each time origin may be specified. If neither is specified, no mto
    calculation occurs. 
    Returns a ([TO x 1]) array of msds and a ([TO x 1]) array of lagtimes
    source: MSD, 6/15/23
    author: Jack Bond
    """

    #set up multiple time origins
    time_origins, lag_idx = _mto(times,max_lag=max_lag,orig_num=orig_num,delta=delta)

    #to save memory we loop over time origins and repeatedly add them to the final array
    mto_average = np.zeros(lag_idx)
    for og in time_origins:
        msd_comp = (coords[og:(og+lag_idx)]-coords[og])**2
        
        chords = np.sqrt(msd_comp.sum(axis=-1))
        arcs = chord_to_arc(chords)
        msd_full = chords**2
        
        ptcl_average = msd_full.mean(axis=-1)
        mto_average += ptcl_average/len(time_origins)

    return mto_average, times[:lag_idx]


def mto_msd_part_arc(coords, times, max_lag=None, orig_num = None, delta = None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd in the arclength coordinateper particle up to the given max lagtime.
    MSDs are then averaged together over several subsets of length [TO],
    within the full time-ordered list. The number of, or distance between,
    each time origin may be specified. If neither is specified, no mto
    calculation occurs. 
    Returns a ([TO x N]) array of msds and a ([TO x 1]) array of lagtimes
    source: MSD, 6/15/23
    author: Jack Bond
    """

    #set up multiple time origins
    time_origins, lag_idx = _mto(times,max_lag=max_lag,orig_num=orig_num,delta=delta)

    #to save memory we loop over time origins and repeatedly add them to the final array
    mto_average = np.zeros((lag_idx,coords.shape[1]))
    for og in time_origins:
        msd_comp = (coords[og:(og+lag_idx)]-coords[og])**2
        
        chords = np.sqrt(msd_comp.sum(axis=-1))
        arcs = chord_to_arc(chords)
        msd_full = chords**2
        
        mto_average += msd_full/len(time_origins)
        
    return mto_average, times[:lag_idx]


def bootstrap_msd(msd_part, trials, confidence=95, rng=default_rng()):
    """
    Given particle msds (either mto or not) of shape [T x N], performs bootstrapping to estimate a
    specified confidence interval.
    source: general_analysis, 7/23/22
    author: Alex yeh, Jack Bond
    """
    boot_msd = np.zeros((msd_part.shape[0], trials))
    
    for b in range(trials):
        # get indices with replacement
        boot_idx = rng.integers(0, msd_part.shape[1], msd_part.shape[1])
        # average over msds for each bootstrap trial
        boot_msd[:, b] = np.mean(msd_part[:, boot_idx], axis=-1)
        
    #get confidence intervals
    low = (100 - confidence)/2
    high = 100 - low
    low_bound = np.percentile(boot_msd, low, axis=1)
    high_bound = np.percentile(boot_msd, high, axis=1)
    
    return low_bound, high_bound


#CENTER OF MASS CORRECTIONS

def _com(coords, masses = None, on_sphere=False):
    """
    returns the center of mass trajectory of a set of frames given the mass of
    each particle in the frame.
    if on_sphere, the center of mass point is projected to the surface of the
    sphere which coords sit on.
    source: MSD, 6/15/23
    author: Jack Bond
    """
    pnum = coords.shape[1]
    #compute the center of mass of every frame
    if masses is None:
        masses = np.ones(pnum)
    com = np.average(coords, weights=masses,axis=1)

    if on_sphere:
        rad = np.linalg.norm(coords,axis=-1).mean()
        com_rad = np.linalg.norm(com,axis=-1).mean()
        com = com*rad/com_rad

    return com

def mto_com_msd(coords, times, spherical=False, masses=None, max_lag=None, orig_num = None, delta = None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the center-of-mass msd up to the given max lagtime
    MSDs are then averaged together over several subsets of length [TO],
    within the full time-ordered list. The number of, or distance between,
    each time origin may be specified. If neither is specified, no mto
    calculation occurs. 
    Returns a ([TO x 1]) array of msds and a ([TO x 1]) array of lagtimes
    source MSD, 6/15/23
    author: Jack Bond
    """

    #set up multiple time origins
    time_origins, lag_idx = _mto(times,max_lag=max_lag,orig_num=orig_num,delta=delta)

    #compute center of mass
    com = _com(coords,masses=masses,on_sphere=spherical)

    #to save memory we loop over time origins and repeatedly add them to the final array
    mto_average = np.zeros(lag_idx)
    for og in time_origins:
        msd_comp = (com[og:(og+lag_idx)]-com[og])**2
        msd_full = msd_comp.sum(axis=-1)
        mto_average += msd_full/len(time_origins)

    return mto_average, times[:lag_idx]


def bootstrap_com_msd(coords, traj_length, n_subsets, trials, 
        masses = None, subset_size=None, spherical = False,
        confidence=95, rng=default_rng()):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), performs bootstrapping to estimate a
    specified confidence interval for the center of mass motion by randomly taking.
    
    Returns a tuple of three ([TO x N]) arrays of msds and one ([T x 1]) array of lagtimes
    source: MSD, 6/20/23
    author: Jack Bond,
    """

    fnum,pnum,_ = coords.shape

    if subset_size is None:
        subset_size = int(pnum/10)

    #generate n_subsets com trajectories
    com_msds = np.zeros((traj_length,n_subsets))
    for i in range(n_subsets):
        i_start = rng.integers(0,fnum-traj_length)
        rand_point = coords[0,rng.integers(0,pnum)]
        subset_idx = np.argsort(np.linalg.norm(coords[i_start]-rand_point,axis=-1))[:subset_size]
        traj = coords[i_start:(i_start+traj_length)][:,subset_idx]
        
        subset_com = _com(traj,on_sphere=spherical,masses=masses)
        com_msds[:,i] = np.sum((subset_com-subset_com[0])**2,axis=-1)

    low_bound, high_bound = bootstrap_msd(com_msds,trials, confidence=confidence,rng=rng)

    return low_bound, high_bound

#COMPOUND ANALYSES


def mto_msd_weighted_voronoi(coords, times, vor, max_lag=None, orig_num=None, delta = None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), and a list of
    coordination numbers ([T x N]), computes the msd per particle up to the
    given max lagtime.
    MSDs are then weighted by the time each particle spends n-coordinated
    over several subsets of length [TO] within the full time-ordered list,
    before they are averaged together. The number of, or distance between,
    each time origin may be specified. If neither is specified, no mto
    calculation occurs. 
    Returns a tuple of three ([TO x 1]) arrays of msds and one ([TO x 1]) array of lagtimes
    source: MSD, 6/20/23
    author: Jack Bond, Alex Yeh
    see: Vest, Tarjus, Viot, J. Chem. Phys. (2018) https://doi.org/10.1063/1.5027389
    """


    #compute normalization factors based on the number of particles per frame with
    #each coordination number
    norm5 = np.sum(vor==5,axis=-1).mean()
    norm6 = np.sum(vor==6,axis=-1).mean()
    norm7 = np.sum(vor==7,axis=-1).mean()

    #set up multiple time origins
    time_origins, lag_idx = _mto(times,max_lag=max_lag,orig_num=orig_num,delta=delta)

    #to save memory we loop over time origins and repeatedly add them to the final array
    msd5 = np.zeros(lag_idx)
    msd6 = np.zeros(lag_idx)
    msd7 = np.zeros(lag_idx)
    for og in time_origins:
        msd_comp = (coords[og:(og+lag_idx)]-coords[og])**2
        msd_full = msd_comp.sum(axis=-1)

        #calculating the fraction spent defected per time window, per time origin
        time_divide = np.array([np.arange(lag_idx) for _ in range(coords.shape[1])]).T
        time_divide[0,:] = -1 # just to fix divide by zero errors
        weight5 = np.cumsum(vor[og:(og+lag_idx)]==5,axis=0)/time_divide
        weight6 = np.cumsum(vor[og:(og+lag_idx)]==6,axis=0)/time_divide
        weight7 = np.cumsum(vor[og:(og+lag_idx)]==7,axis=0)/time_divide

        #performing the weighted average using a fixed 
        weighted5 = np.sum(msd_full*weight5,axis=-1)/norm5
        weighted6 = np.sum(msd_full*weight6,axis=-1)/norm6
        weighted7 = np.sum(msd_full*weight7,axis=-1)/norm7

        msd5 += weighted5/len(time_origins)
        msd6 += weighted6/len(time_origins)
        msd7 += weighted7/len(time_origins)

    return (msd5, msd6, msd7), times[:lag_idx]

    #idk this was kinda working earlier
    #msd7 = np.einsum("mtn, mn -> mt", msd_full, weight7).mean(axis=0)


def find_DL(msd, lagtime, window=100, msd_func = lambda t, D: 4*D*t):
    """
    source:MSD 3/14/23
    author: Jack Bond
    """
    ts = []
    Ds = []
    dDs = []
    for i,t in enumerate(lagtime[:-1*window]):
        ts.append(np.mean(lagtime[i:i+window]))
        fit_x = lagtime[i:i+window]-lagtime[i]
        fit_y = msd[i:i+window] - msd[i]
        d, cov = curve_fit(msd_func,fit_x,fit_y,p0=[1e-1])
        Ds.append(d[0])
        dDs.append(np.sqrt(cov[0,0]))

    Ds = np.array(Ds)
    dDs = np.array(dDs)
    ts = np.array(ts)
    D_diff = Ds[1:]-Ds[:-1]
    tol = np.std(D_diff)
    if not np.any(D_diff>tol):
        tol = 0
    i_cross = np.where(D_diff>tol)[0][0]

    return ts, Ds, dDs, i_cross


#Alex's old version for benchmarking
def _alex_old_mto_msd(coords, max_lag, skips=None):
    """
    source: general_analysis, 7/23/22
    author: Alex yeh
    """
    #set up multiple time origins
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    time_origins = np.linspace(0,total_steps-max_lag,orig_num).astype(int)
    final_step = time_origins[-1]+max_lag
    assert final_step<=total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    #compute msds
    msd_comp = np.zeros((max_lag, 3))
    for t in range(max_lag):
        for tstart in time_origins:
            tend = tstart + t
            allmsd = (coords[tend]-coords[tstart])**2 #Nx3
            msd_comp[t] += np.sum(allmsd,axis=0) #3
    return msd_comp/(pnum*orig_num)


def _alex_old_vor_weight(coords, coord_nums, max_lag, skips=None):
    """
    ->edited to only do fraction 5-fold to make it more readable and troubleshootable
    source: general_analysis, 7/23/22
    author: Alex yeh
    """
    #set up multiple time origins
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    time_origins = np.linspace(0,total_steps-max_lag,orig_num).astype(int)
    final_step = time_origins[-1]+max_lag
    assert final_step<=total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    #compute weighted msds
    msd5 = np.zeros((max_lag))
    for tstart in time_origins:
        skip_coord_nums = coord_nums[tstart:(tstart+max_lag)]
        mask5 = skip_coord_nums==5
        fiv_coord_counts = np.cumsum(mask5, axis=0) #sum over time
        for t in range(0, max_lag-1):
            tend = tstart + (t + 1)
            fiv_frac = fiv_coord_counts[t]/(t+1)
            msd = (coords[tend] - coords[tstart])**2
            if mask5[t].any():
                msd5[t+1] += np.sum(fiv_frac*np.sum(msd, axis=-1))/mask5[t].sum()

    norm = orig_num
    return msd5/norm


#FOR TESTING: VERY MUTABLE
if __name__=="__main__":

    from timeit import default_timer as timer
    import matplotlib.pyplot as plt
    import json
    
    start = timer()

    #load data from a preprocessed simulation
    config = json.load(open('config.json','r'))

    coords = np.load('example_datapts.npy')
    times = np.load('times.npy')*config['arg']['xxxtimestepxxx']
    #coords = np.load('example_datapts_minim.npy')
    #times = np.load('times.npy')
    
    fnum,pnum,_= coords.shape
    vor = np.load('example_vor_coord.npy')
    
    
    # max_lag = 50 #tau units
    # or_n = 100 #num origins
    # #delta = 2
    # s = timer()
    # msd, lag = mto_msd(coords,times, max_lag=max_lag,orig_num=or_n)
    # print(f"{timer()-s:.5f}s new msd")
    # s = timer()
    # msd_vor_tuple, _ = mto_msd_weighted_voronoi(coords,times, vor, max_lag=max_lag,orig_num=or_n)
    # print(f"{timer()-s:.5f}s new vor weighting")

    # #match the mto parameters to the old versions
    # max_lag_idx = int(max_lag/(config['arg']['xxxnsnapxxx']*config['arg']['xxxtimestepxxx']))
    # skips = int(len(times)/or_n) #delta
    # s = timer()
    # msd_old = _alex_old_mto_msd(coords, max_lag_idx,skips=skips).sum(axis=-1)
    # print(f"{timer()-s:.5f}s old msd")
    # s = timer()
    # msd5 = _alex_old_vor_weight(coords, vor, max_lag_idx, skips=skips)
    # print(f"{timer()-s:.5f}s old vor weighting")
    # lagtimes = times[:max_lag_idx]


    
    # fig,ax = plt.subplots()
    
    # ax.plot(lag,lag, color = 'k')
    # ax.plot(lagtimes,msd_old, label = 'ens (old)',lw=1.5)
    # ax.plot(lag,msd, label = 'ens (new)',ls='--')
    # ax.plot(lagtimes,msd5, label = 'frac5 (old)',lw=1.5)
    # ax.plot(lag,msd_vor_tuple[0], label = 'frac5 (new)',ls='--')

    # print(msd_vor_tuple[0])

    # ax.legend()
    # ax.set_ylim([msd.min(),1.1*msd.max()])
    # fig.savefig('temp_vor_testing.jpg',bbox_inches='tight')


    fig,ax = plt.subplots(figsize=(3.25,3.25),dpi=600)
    ax.set_xlabel("$t/\\tau$",fontsize=12)
    ax.set_ylabel("$<\delta r^2>/(2a)^2$",fontsize=12)

    #ensemble msd and confidence itnerval
    msd_part, lag = mto_msd_part(coords,times, max_lag=100, orig_num=100)
    msd_ens = msd_part.mean(axis=-1)
    low_bound, high_bound = bootstrap_msd(msd_part, 1000)

    ax.plot(lag, msd_ens,color='k',ls=':')
    ax.fill_between(lag,low_bound,high_bound,facecolor='grey',alpha=0.6,edgecolor='k',lw=0.3)
    ax.plot(lag, (low_bound+high_bound)/2,color='grey',ls=':')

    #com confidence interval
    low_bound_com, high_bound_com = bootstrap_com_msd(coords, len(lag),300,1000, spherical=False,subset_size=100)
    
    ax.fill_between(lag,low_bound_com,high_bound_com,facecolor='blue',alpha=0.6,edgecolor='blue',lw=0.3)
    ax.plot(lag, (low_bound_com+high_bound_com)/2,color='blue',ls=':')
    print(high_bound_com[-1])

    #for flat-cases we can just calcilate the com motiom
    #msd_com, _ = mto_com_msd(coords,times, max_lag=100, orig_num=100, spherical=False)
    #ax.plot(lag, msd_com,color='red',ls=':')

    #input diffusivity for visual reference
    ax.plot(lag, sphere_msd(lag,0.25,shell_radius=np.linalg.norm(coords).mean()), color='k',lw=0.5)

    ax.set_ylim(0,1.1*msd_ens.max())
    fig.savefig('temp.jpg',bbox_inches='tight')




    end = timer()
    print(f"{end-start:.5f}s runtime")
