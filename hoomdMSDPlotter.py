# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Plots the mean squared displacement, and several variations thereof for a lammps run

Also performs fits to find the diffusivities

@author: Jack Bond, Alex Yeh
"""

import glob, os, sys, json

import numpy as np
from numpy.random import default_rng
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from main_lib.FileHandling import read_infile, read_dump, read_thermo, get_thermo_time, dumpDictionaryJSON
from main_lib.UnitConversions import kb, getAEff
from main_lib.MSD import *

import gsd.hoomd


if __name__=="__main__":

    from timeit import default_timer as timer

    allstart = timer()

    # load data
    path = './'
    print(path)
    #%% read dump file
    reset = False

    '''reading infiles'''
    start = timer()
    
    hoomd_out = gsd.hoomd.open("traj.gsd")
    multiple = np.array([f.particles.position for f in hoomd_out])

    config = json.load(open('config.json','r'))
    eta = config['eta']
    
    pnum = multiple.shape[1]
    D0 = 1

    Lbox = np.sqrt( pnum*np.pi/4 * 2/np.sqrt(3) / eta )

    dt = 0.1
    times = dt*np.arange(len(hoomd_out))

    title = fr"N: {pnum}, $\eta$: {eta:.3f}"
    

    end = timer()
    interval = end - start
    print(f"read and process files {interval:.2f}s")
    start = timer()
        
    #%% calculate msd
    msd_time_scale = int( Lbox**2/(0.1*D0) )

    s = 100 - 50*(pnum<300)- 25*(pnum<50) #- 15*(pnum<10)

    msd_comp = mto_msd_pbc(multiple, msd_time_scale, skips = s, box_length=Lbox)
    msd_part = mto_msd_part(multiple, msd_time_scale, skips = s)
    msd = msd_comp.sum(axis=-1)
    msd_times = times[:msd_time_scale]
    np.savetxt(path+f'msd_{msd_time_scale}frames.txt',
               (msd_times, msd), header='tau msd[2a]^2')

    #%% get bootstrap error
    trials = 1000
    rng = default_rng()
        
    #get confidence intervals
    msd_ci = bootstrap_mto_msd(msd_part, trials, rng=rng)

    def msd_func(t,D0):
        return 4*D0*t
    
    diff_coef, diff_cov = curve_fit(msd_func, msd_times, msd, p0=[1e-1])
    theo = msd_func(msd_times, D0)

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
    ax.set_ylim([0, 1.1*msd_func(msd_times[-1], *diff_coef)])

    ax.set_title(title)

    ax.legend()
    fig.savefig(path+"msd.jpg", bbox_inches='tight')

    short_time = (0.5**2)/(4*D0)
    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, short_time])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, 1.1*4*D0*short_time])
    ax.set_title("Self-Diffusion over Particle Radius")

    ax.legend()
    fig.savefig(path+"msd_short.jpg", bbox_inches='tight')

    print(f"msd calculation {end - start:.2f}s")

    output = {
            'D_0_fit': diff_coef[0],
            'D_0': D0,
    }

    dumpDictionaryJSON(output, 'diffusion')

    allend = timer()
    print(f"full runtime {allend-allstart:.2f}s")
