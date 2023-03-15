# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Contains lots of methods to compute msds and variatons thereof for particles on flat and curved surfaces.

@author: Jack Bond, Alex Yeh
"""

import numpy as np
from numpy.random import default_rng
from scipy.optimize import curve_fit
from Correlation import theta1,theta2

"""
Theoretical expression for the diffusive mean-squared displacement on a spherical surface
source: general_analysis, 7/23/22
author: Alex yeh
"""
def sphere_msd(taus, damp, shell_radius = 10):
    return 2*(shell_radius**2)*(1-np.exp(-2*damp*taus*shell_radius**-2))


"""
source MSD, 2/9/23
author: Jack Bond
"""
def minimum_image(coords, wraps, basis):
    """
    uses the minumum image convention to correctly account for perodic boundary conditions when calculating coordinates by using the basis vectors of the periodic box.
    """
    disp = np.einsum("ij,anj->ani",basis,wraps)
    return coords + disp


"""
source: general_analysis, 7/23/22
author: Alex yeh
"""
def mto_msd(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd up to the given max_step, defaulting to the maximum number of 
    non-overlapping multiple time origins. Overlapping time orgins can be
    given by specifying a skip param less than 2*max_lag.
    Returns a [T x 3] array of msds
    """

    #set up multiple time origins
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    print(f"time origins: {orig_num}")
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
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")
    return msd_comp/(pnum*orig_num)


"""
source: general_analysis, 7/23/22
author: Alex yeh
"""
def mto_msd_part(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. Overlapping time orgins 
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
    """

    #set up the multiple time origins
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    time_origins = np.linspace(0,total_steps-max_lag,orig_num).astype(int)
    final_step = time_origins[-1]+max_lag
    assert final_step<=total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    #compute msd
    msd = np.zeros((max_lag, pnum, 3))
    
    for t in range(max_lag):
        for tstart in time_origins:
            tend = tstart + t
            msd[t] += (coords[tend] - coords[tstart])**2
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")
    
    return msd/orig_num


"""
source: general_analysis, 7/23/22
author: Jack Bond
"""
def mto_msd_arc(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd in the arclength coordinate up to the given max_step, defaulting
    to the maximum number of non-overlapping multiple time origins.
    Overlapping time orgins can be given by specifying a skip param less than
    2*max_lag. Returns a [T x 3] array of msds
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
    msd_w = np.zeros((max_lag, 1))
    shell_radius = np.linalg.norm(coords, axis=-1).mean()
    for t in range(max_lag):
        for tstart in time_origins:
            tend = tstart + t
            allmsd = (coords[tend]-coords[tstart])**2 #Nx3

            arg = np.sqrt(allmsd.sum(axis=-1))/(2*shell_radius) #N
            arg[arg>1] = 1
            
            msd_w[t] += np.sum((2*shell_radius*np.arcsin(arg))**2) #1
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")
    return msd_w/(pnum*orig_num)


"""
source MSD, 2/9/23
author: Jack Bond
"""
def mto_com_msd(coords,max_lag,skips=None, masses=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the center-of-mass msd up to the given max_step, defaulting to the maximum
    number of non-overlapping multiple time origins. Overlapping time orgins
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x 3] array of msds, as well as the center of mass coordinates.
    """

    #set up the multiple time origins
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos

    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    time_origins = np.linspace(0,total_steps-max_lag,orig_num).astype(int)
    final_step = time_origins[-1]+max_lag
    assert final_step<=total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    #compute the center of mass of every frame
    if masses is None:
        masses = np.ones(pnum)
    com = np.einsum("n,fni->fi",masses,coords)/masses.sum()
    
    #compute msd
    msd_comp = np.zeros((max_lag, 3))

    for t in range(max_lag):
        for tstart in time_origins:
            tend = tstart + t
            msd_comp[t] +=  (com[tend]-com[tstart])**2#3
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")
    return msd_comp/(orig_num), com

"""
source MSD, 2/14/23
author: Jack Bond
"""
def sector_msd(coords,masses=None,theta_c=None,phi_c=None,subtended_halfangle=theta1/2,shell_radius=None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes the center-of-mass msd for a subset of particles within a subtended angle of some point on the sphere (given by theta and phi). Returns a [T x 3] array of msds
    """

    #picking the central point around which we track particles
    if theta_c is None:
        theta_c = np.pi*np.random.random()
    if phi_c is None:
        phi_c = 2*np.pi*np.random.random()
    if shell_radius is None:
        shell_radius = np.linalg.norm(coords,axis=-1).mean()
    
    central_vec = np.array([shell_radius*np.sin(theta_c)*np.cos(phi_c),shell_radius*np.sin(theta_c)*np.sin(phi_c),shell_radius*np.cos(theta_c)])

    fnum, pnum, _ = coords.shape

    unit_vecs = np.array([c/np.linalg.norm(c) for c in coords[0]])
    unit_center = central_vec/np.linalg.norm(central_vec)
    args = np.einsum("ni,i->n",unit_vecs,unit_center)

    idx = np.abs(np.arccos(args)) < subtended_halfangle

    subset = coords[:,idx]
    
    #compute the center of mass of every frame
    if masses is None:
        masses = np.ones(pnum)

    if masses[idx].sum()==0:
        com_adj = np.zeros((fnum,3))
        rel_rad = np.zeros(fnum)
    else:
        com = np.einsum("n,fni->fi",masses[idx],subset)/(masses[idx].sum())
        #adjust the com to keep it on the spehere surface
        com_adj = shell_radius*np.array([f/np.linalg.norm(f) for f in com])
        rel_rad = np.linalg.norm(com,axis=-1)

    disp_ens = subset-subset[0]
    disp_com = com_adj-com_adj[0]

    msd_ens = np.mean((disp_ens)**2, axis=1)
    msd_com = (disp_com)**2

    return msd_ens, msd_com, rel_rad, idx, central_vec

"""
source MSD, 2/9/23
author: Jack Bond
"""
def mto_sector_msd(coords,max_lag,skips=None, masses=None,theta_c=None,phi_c=None,subtended_halfangle=theta1/2,shell_radius=None):
    """
    Given a set of T timesteps of N particles ([T x N x 3]), computes the ensemble and center-of-mass msd for a subset of particles within a subtended angle of some point on the sphere (given by theta and phi), up to the given max_step, defaulting to the maximum number of non-overlapping multiple time origins. Overlapping time orgins can be given by specifying a skip param less than 2*max_lag. Returns a [T x 3] array of msds
    """

    #picking the central point around which we track particles
    if theta_c is None:
        theta_c = np.pi*np.random.random()
    if phi_c is None:
        phi_c = 2*np.pi*np.random.random()
    if shell_radius is None:
        shell_radius = np.linalg.norm(coords,axis=-1).mean()
    
    central_vec = np.array([shell_radius*np.sin(theta_c)*np.cos(phi_c),shell_radius*np.sin(theta_c)*np.sin(phi_c),shell_radius*np.cos(theta_c)])

    # setting up multiple time origins
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    time_origins = np.linspace(0,total_steps-max_lag,orig_num).astype(int)
    final_step = time_origins[-1]+max_lag
    assert final_step<=total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    if masses is None:
        masses = np.ones(pnum)

    msd_com = np.zeros((max_lag, 3))
    msd_ens = np.zeros((max_lag, 3))
    md_rad = np.zeros((max_lag, 1))
    mean_n = 0
    
    for tstart in time_origins:
        
        #define subset of tracked particles by taking the arcos of the dot product between each point and the central vector
        #notably, once we choose particles at each time origin, we keep them throughout the whole of each time window.
        unit_vecs = np.array([c/np.linalg.norm(c) for c in coords[tstart]])
        unit_center = central_vec/np.linalg.norm(central_vec)
        args = np.einsum("ni,i->n",unit_vecs,unit_center)
        
        idx = np.abs(np.arccos(args)) < subtended_halfangle
        
        #if no particles are in the sector we can't use this time origin
        if np.sum(idx)==0:
            orig_num-=1
            continue
        mean_n += np.sum(idx)
        
        #find the center of mass of those particles
        if masses[idx].sum()==0:
            com_adj = np.zeros((total_steps,3))
            rel_rad = np.zeros(total_steps)
        else:
            com = np.einsum("n,fni->fi",masses[idx],coords[:,idx,:])/(masses[idx].sum())
            #adjust the com to keep it on the spehere surface
            com_adj = shell_radius*np.array([f/np.linalg.norm(f) for f in com])
            rel_rad = np.linalg.norm(com,axis=-1)
        
        for t in range(max_lag):
            tend = tstart + t
            msd_com[t] += (com_adj[tend]-com_adj[tstart])**2 #3
            msd_ens[t] += np.mean((coords[tend,idx,:]-coords[tstart,idx,:])**2,axis=0) #3
            md_rad[t] += (rel_rad[tend]-rel_rad[tstart]) #1
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")

    #if no particles ever get recorded in the sector, return zeros
    if orig_num==0:
        return np.zeros(max_lag,3), np.zeros(max_lag,pnum,3), np.zeros(max_lag), np.zeros(max_lag), central_vec
    
    return msd_com/orig_num, msd_ens/orig_num, md_rad/orig_num, mean_n/orig_num, central_vec


"""
source: MSD, 2/2/23
author: Jack Bond
"""
def mto_msd_hex(coords, coord_nums, max_lag, skips=None,min_six_frac=0.90):
    """Given a set of T timesteps of N particles ([T x N x 3]), determines
    which particles spend most of the runtime as 6-coordinated, computes 
    the msd per each of these particles up to the given max_step, defaulting to the maximum number of non-overlapping multiple time origins.
    Overlapping time orgins  can be given by specifying a skip param less than 2*max_lag. Returns a [T] array of ensemble-averaged msds
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
    

    #select 6-fold ptcls and compute msds
    msd = np.zeros((max_lag))
    num_hex = np.zeros((max_lag))
    
    for tstart in time_origins:
        skip_coord_nums = coord_nums[tstart:(tstart+max_lag)]
        mask6 = skip_coord_nums==6
        six_coord_counts = np.cumsum(mask6, axis=0) #sum over time
        # print('-'*40)
        # print(six_coord_counts)
        for t in range(0, max_lag-1):
            tend = tstart + (t + 1)
            six_frac = six_coord_counts[t]/(t+1)
            hex_subset = six_frac >= min_six_frac
            msd_subset = (coords[tend,hex_subset,:] - coords[tstart,hex_subset,:])**2
            if np.any(hex_subset):
                msd[t+1] += np.mean(msd_subset,axis=0).sum() #ensemble average
                num_hex[t+1] += np.sum(hex_subset)


    norm = orig_num
    return msd/norm, num_hex/norm

"""
source: general_analysis, 7/23/22
author: Alex yeh
"""
def mto_msd_part_Vcweight(coords, coord_nums, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. Msds are weighted by 
    time spent per coordination number. Overlapping time orgins 
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
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
    msd6 = np.zeros((max_lag))
    msd7 = np.zeros((max_lag))
    
    for tstart in time_origins:
        skip_coord_nums = coord_nums[tstart:(tstart+max_lag)]
        mask5 = skip_coord_nums==5
        mask6 = skip_coord_nums==6
        mask7 = skip_coord_nums==7
        fiv_coord_counts = np.cumsum(mask5, axis=0) #sum over time
        six_coord_counts = np.cumsum(mask6, axis=0)
        sev_coord_counts = np.cumsum(mask7, axis=0)
        # print('-'*40)
        # print(six_coord_counts)
        for t in range(0, max_lag-1):
            tend = tstart + (t + 1)
            fiv_frac = fiv_coord_counts[t]/(t+1)
            six_frac = six_coord_counts[t]/(t+1)
            sev_frac = sev_coord_counts[t]/(t+1)
            msd = (coords[tend] - coords[tstart])**2
            if mask5[t].any():
                msd5[t+1] += np.sum(fiv_frac*np.sum(msd, axis=-1))/mask5[t].sum()
            if mask6[t].any():
                msd6[t+1] += np.sum(six_frac*np.sum(msd, axis=-1))/mask6[t].sum()
            if mask7[t].any():
                msd7[t+1] += np.sum(sev_frac*np.sum(msd, axis=-1))/mask7[t].sum()
            # print(six_coord_frac)
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4} ")
    norm = orig_num
    return msd5/norm, msd6/norm, msd7/norm

"""
source: MSDPlotter, 1/19/23
author: Jack Bond
"""
def mto_msd_part_Scarweight(coords, scar_nums, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. MSDs are weighted by time
    spent in various classifications of defect clusters. Overlapping time
    orgins can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
    """
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    time_origins = np.linspace(0,total_steps-max_lag,orig_num).astype(int)
    final_step = time_origins[-1]+max_lag
    assert final_step<=total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    # origins = np.arange(orig_num)*skips
    # print("(j,t) | tstart | tend | diff ")
    msdelse = np.zeros((max_lag))
    msdzero = np.zeros((max_lag))
    
    for tstart in time_origins:
        skip_scar_nums = scar_nums[tstart:(tstart+max_lag)]
        maskelse = (skip_scar_nums !=0)*(skip_scar_nums != None)
        maskzero = skip_scar_nums==0
        else_coord_counts = np.cumsum(maskelse, axis=0) #sum over time
        zero_coord_counts = np.cumsum(maskzero, axis=0)
        # print('-'*40)
        # print(six_coord_counts)
        for t in range(0, max_lag-1):
            tend = tstart + (t + 1)
            else_frac = else_coord_counts[t]/(t+1)
            zero_frac = zero_coord_counts[t]/(t+1)
            msd = (coords[tend] - coords[tstart])**2
            if maskelse[t].any():
                msdelse[t+1] += np.sum(else_frac*np.sum(msd, axis=-1))/maskelse[t].sum()
            if maskzero[t].any():
                msdzero[t+1] += np.sum(zero_frac*np.sum(msd, axis=-1))/maskzero[t].sum()
            # print(six_coord_frac)
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4} ")
    norm = orig_num
    return msdelse/norm, msdzero/norm

"""
source: general_analysis, 7/23/22
author: Alex yeh
"""
def bootstrap_mto_msd(msd_part, trials, 
                      skips=None, confidence=95, rng=default_rng()):
    summed_msd = np.sum(msd_part, axis=-1)
    ave_msd = np.average(summed_msd, axis=-1)
    boot_msd = np.zeros((msd_part.shape[0], trials))
    
    for b in range(trials):
        # get indices with replacement
        boot_idx = rng.integers(0, msd_part.shape[1], msd_part.shape[1])
        # average over msds for each bootstrap trial
        boot_msd[:, b] = np.average(summed_msd[:, boot_idx], axis=-1)
        
    #get confidence intervals
    msd_ci = np.zeros((2, msd_part.shape[0]))
    low = (100 - confidence)/2
    high = 100 - low
    
    msd_ci[0] = ave_msd - np.percentile(boot_msd, low, axis=1)
    msd_ci[1] = np.percentile(boot_msd, high, axis=1) - ave_msd
    return msd_ci

"""
source:MSD 3/14/23
author: Jack Bond
"""
def find_DL(lagtime, msd, dim=2, window=100):

    msd_func = lambda t, D: 2*dim*D*t
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
    i_cross = np.where(D_diff>0)[0][0]

    return ts, Ds, dDs, i_cross

if __name__=="__main__":

    from timeit import default_timer as timer

    start = timer()


    end = timer()
    print(f"{end-start}s runtime")
