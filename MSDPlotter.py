# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Plots the mean squared displacement, and several variations thereof for a lammps run

Also performs fits to find the diffusivities

@author: Jack Bond, Alex Yeh
"""

import glob, os
import numpy as np

from FileHandling import read_infile, read_dump, read_thermo, get_thermo_time
from numpy.random import default_rng
import matplotlib.pyplot as plt

from timeit import default_timer as timer

from scipy.optimize import curve_fit

def sphere_msd(taus, damp, shell_radius = 10):
    return 2*(shell_radius**2)*(1-np.exp(-2*damp*taus*shell_radius**-2))

def mto_msd(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd up to the given max_step, defaulting to the maximum number of 
    non-overlapping multiple time origins. Overlapping time orgins can be
    given by specifying a skip param less than 2*max_lag.
    Returns a [T x 3] array of msds
    """
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos

    shell_radius = np.linalg.norm(coords, axis=-1).mean()
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    final_step = (orig_num-1)*skips + (max_lag-1) #last necessary timestep
    assert final_step<total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    
    # origins = np.arange(orig_num)*skips
    # print("(j,t) | tstart | tend | diff ")
    msd_comp = np.zeros((max_lag, 3))
    msd_w = np.zeros((max_lag, 1))
    
    for t in range(max_lag):
        for tstart in range(0, orig_num*skips, skips):
            tend = tstart + t
            allmsd = (coords[tend]-coords[tstart])**2 #Nx3

            msd_comp[t] += np.sum(allmsd,axis=0) #3

            arg = np.sqrt(allmsd.sum(axis=-1))/(2*shell_radius) #N
            arg[arg>1] = 1
            msd_w[t] += np.sum((2*shell_radius*np.arcsin(arg))**2) #1
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")
    return msd_comp/(pnum*orig_num), msd_w/(pnum*orig_num)

def mto_msd_part(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. Overlapping time orgins 
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
    """
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    final_step = (orig_num-1)*skips + (max_lag-1) #last necessary timestep
    assert final_step<total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    # origins = np.arange(orig_num)*skips
    # print("(j,t) | tstart | tend | diff ")
    msd = np.zeros((max_lag, pnum, 3))
    
    for t in range(max_lag):
        for tstart in range(0, orig_num*skips, skips):
            tend = tstart + t
            msd[t] += (coords[tend] - coords[tstart])**2
            # print(f"({tstart},{t})   {tstart: ^6} | {tend: ^4} | {tend-tstart: ^4}")
    
    return msd/orig_num

def mto_msd_part_Vcweight(coords, coord_nums, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. Overlapping time orgins 
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
    """
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    final_step = (orig_num-1)*skips + (max_lag-1) #last necessary timestep
    assert final_step<total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    # origins = np.arange(orig_num)*skips
    # print("(j,t) | tstart | tend | diff ")
    msd5 = np.zeros((max_lag))
    msd6 = np.zeros((max_lag))
    msd7 = np.zeros((max_lag))
    
    for tstart in range(0, orig_num*skips, skips):
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

def mto_msd_part_Scarweight(coords, scar_nums, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. Overlapping time orgins 
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
    """
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    final_step = (orig_num-1)*skips + (max_lag-1) #last necessary timestep
    assert final_step<total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    # origins = np.arange(orig_num)*skips
    # print("(j,t) | tstart | tend | diff ")
    msdelse = np.zeros((max_lag))
    msdzero = np.zeros((max_lag))
    
    for tstart in range(0, orig_num*skips, skips):
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

if __name__=="__main__":

    allstart = timer()

    # load data
    path = './'
    print(path)
    #%% read dump file
    reset = False

    '''reading infiles'''

    start = timer()
    infile = glob.glob(path+'*.in')
    assert len(infile) == 1, "need to have one specified input file"

    lammps_params = read_infile(infile[0])
    thermo_header, thermo = read_thermo(path+'log.lammps')
    time_str = get_thermo_time(path+'log.lammps')

    filename = path+'out.dump'
    if os.path.exists(path+'datapts.npy'):
        multiple = np.load(path+'datapts.npy')
        ts = np.load(path+'times.npy')
    else:
        multiple, ts = read_dump(filename)
        np.save(path+'datapts.npy',multiple)
        np.save(path+'times.npy',ts)

    import json
    config = json.load(open('config.json','r'))
    params = config['params']

    from UnitConversions import kb
    a_hc = params['particle_radius']
    kT = params['temperature']*kb
    visc = params['viscosity']
    D_SI = kT/(6*np.pi*visc*a_hc)

    pnum = multiple.shape[1]
    # dt = lammps_params['timestep']*tau # [s]

    damp = lammps_params['damp']
    mass = config['arg']['xxxmassxxx']
    temp = config['arg']['xxxtempxxx']
    D0 = temp*damp/mass

    tau_D = 1/(4*D0)
    tau_D_SI = (a_hc**2)/(D_SI)

    #kappa = lammps_params['kappa_2a']/(2*a_hc)
    #bpp = lammps_params['bpp']
    dt = lammps_params['timestep']

    times = ts*dt
    a_eff = a_hc#int_a_eff(a_hc, bpp, kappa)

    if lammps_params['rad'] is None:
        shell_radii = np.linalg.norm(multiple,axis=-1).mean(axis=-1)
        shell_radius = shell_radii.min() #get stand in value for rough approximation
        all_eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radii**2)
        eta_eff = all_eta_eff.max() #get stand in value for rough approximation
        title = (f"N: {pnum}, "
                 +f"R: {shell_radii.max():0.3f}-{shell_radii.min():0.3f} "
                 +r"[2a], $\eta_{eff}$:" 
                 + f"{all_eta_eff.max():0.3f}-{all_eta_eff.min():0.3f}")
    else:
        shell_radius = lammps_params['rad']
        eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radius**2)
        title = (f"N: {pnum}, "
                 +f"R: {shell_radius:0.3f} "
                 +r"[2a], $\eta_{eff}$:" 
                 + f"{eta_eff:0.3f}")

    end = timer()
    interval = end - start
    print(f"read and process files {interval:.2f}s")

    #%% obtaining theoretical diffusivities
    start = timer()
    taus = thermo[:,0]-thermo[:,0].min()
    all_taus = np.linspace(0, thermo[:,0].max(), num=150)
    theo = sphere_msd(all_taus, D0, shell_radius)

    msd_func = lambda x, D0: sphere_msd(x, D0, shell_radius=shell_radius)
        
    totmsd_coef, totmsd_cov = curve_fit(msd_func, taus, thermo[:,-1], p0=[1e-3])
        
    #%% calculate msd
    msd_time_scale = 750
    msd_comp, msd_w = mto_msd(multiple, msd_time_scale)
    msd_part = mto_msd_part(multiple, msd_time_scale)
    msd = msd_comp.sum(axis=-1)
    msd_times = times[:msd_time_scale]
    np.savetxt(path+f'msd_{msd_time_scale}frames.txt',
               (msd_times, msd), header='tau msd[2a]^2')

    #%% get bootstrap error
    trials = 1000
    rng = default_rng()
        
    #get confidence intervals
    msd_ci = bootstrap_mto_msd(msd_part, trials, rng=rng)

    diff_coef, diff_cov = curve_fit(msd_func, msd_times, msd, p0=[1e-1])
    theo = sphere_msd(msd_times, D0, shell_radius)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(msd_times, msd, label='mto msd')
    #ax.plot(msd_times, msd_w, label='mto msd arclength')
    ax.fill_between(msd_times, msd-msd_ci[0], msd+msd_ci[1],
                    alpha=0.3, label='95% bootstrap ci')
    ax.plot(msd_times, theo, color='k', ls=':', label=f'D={D0:0.1e}')
    ax.plot(msd_times, msd_func(msd_times, *diff_coef), 
            color='C0', ls='-.',
            label=f'D={diff_coef[0]:0.3f} (fit)')

    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, msd_times[-1]])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, min(1.1*msd_func(msd_times[-1], *diff_coef),1.2*2*shell_radius**2)])

    ax.plot(msd_times, np.ones(msd_times.size)*2*shell_radius**2,ls='-',label=r'$2R^2$')
    #ax.plot(msd_times, np.ones(msd_times.size)*np.pi*shell_radius**2,ls='-',label=r'$\pi R^2$')

    ax.set_title(title)

    ax.legend()
    fig.savefig(path+"msd.jpg", bbox_inches='tight')
    # ax.plot(thermo[:msd_time_scale,0], thermo[:msd_time_scale,-1], label='lammps msd')
    # fig.savefig(path+"mto_msd_comparison.jpg", bbox_inches='tight')

    short_time = 5*damp
    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, short_time])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, 1.1*4*D0*short_time])
    ax.set_title("Self-Diffusion over Damping Timescale")

    ax.legend()
    fig.savefig(path+"msd_damp.jpg", bbox_inches='tight')

    short_time = (0.5**2)/(4*D0)
    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, short_time])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, 1.1*4*D0*short_time])
    ax.set_title("Self-Diffusion over Particle Radius")

    ax.legend()
    fig.savefig(path+"msd_short.jpg", bbox_inches='tight')

    #if the voronoi tesselation is already done we'll do the charge-weighted msd too
    if os.path.exists(path+'vor_coord.npy'):
        
        coordination = np.load(path+'vor_coord.npy')
        msd5,msd6,msd7 = mto_msd_part_Vcweight(multiple, coordination, msd_time_scale)

        fig, ax = plt.subplots()
        ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
        ax.set_xlabel("[$\\tau$]", fontsize=12)
        ax.plot(msd_times, msd, label='overall', color='k', zorder=5)
        ax.plot(msd_times, msd5, label='5', color='red',lw=0.6,ls='--')
        ax.plot(msd_times, msd6, label='6', color='gray',lw=0.6,ls='--')
        ax.plot(msd_times, msd7, label='7', color='green',lw=0.6,ls='--')
        ax.set_title(title)
        ax.legend()

        fig.savefig(f"./msd_charge.jpg", bbox_inches='tight')

    end = timer()
    print(f"msd calculation {end - start:.2f}s")

    import json
    from FileHandling import dumpDictionaryJSON
    config = json.load(open('config.json', 'r'))

    output = {
            'D_0_fit': diff_coef[0],
            'D_0': D0,
            'D_SI': D_SI,
            'a_hc_SI': a_hc,
            'tau_SI': tau_D_SI/tau_D,
            'D_SI_conv': diff_coef[0]*(2*a_hc)**2/(tau_D_SI/tau_D)
    }

    dumpDictionaryJSON(output, 'diffusion')

    allend = timer()
    print(f"full runtime {allend-allstart:.2f}s")
