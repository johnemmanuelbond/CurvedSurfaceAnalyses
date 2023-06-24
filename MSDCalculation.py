# -*- coding: utf-8 -*-
"""
Created on Fri, Jun 23, 2023

Uses the MSD module to run a baseline diffusivity calculation.
Corrects the MSD for lattice diffusion, where particles do not
escape their local environment and instead move as a large clump
Fits corrected MSD to extrapolate a self-diffusion constant for
particles which escape their local environments.

@author: Jack Bond
"""

import os, json
import numpy as np
from scipy.optimize import curve_fit

from main_lib.FileHandling import dump_json
from main_lib.GeometryHelpers import sphere_msd
from main_lib.MSD import mto_com_msd, mto_msd_part
from main_lib.MSD import bootstrap_msd, bootstrap_com_msd

DEFAULT_ARGS = {
    "coord_file": "datapts.npy",
    "time_file": "times.npy",
    "max_lagtime": 150, # in tau units
    "origin_sep": 5, #in tau units
    "n_origins": None,
    "bootstrap_trials": 100,
}

if __name__=="__main__":

    if os.path.exists("MSD_analysis_arguments.json"):
        args = json.load(open('MSD_analysis_arguments.json'))
    else:
        args = DEFAULT_ARGS
        dump_json(args,"MSD_analysis_arguments.json")

    coords = np.load(args['coord_file'])
    times = np.load(args['time_file'])
    fnum,pnum,_ = coords.shape

    flat = np.std(np.linalg.norm(coords[0],axis=-1)) > 0.1

    #define multiple time origin parameters
    assert times[-1] > 2*args['max_lagtime'], "please provide a maximum lagtime less than half the simulation runtime"
    if n_orig is None:
        delta = args['origin_sep']
        orig_num=None
    else:
        delta=None
        orig_num = args['n_origins']

    msd_part, lag = mto_msd_part(coords,times, max_lag=args['max_lagtime'], orig_num=orig_num,delta=delta)
    msd_ens = msd_part.mean(axis=-1)
    
    low_bound, high_bound = bootstrap_msd(msd_part, args['bootstrap_trials'])
    dmsd_ens = np.max([msd_ens-low_bound,high_bound-msd_ens],axis=0)

    #calculate lattice diffusion by tracking the center of mass
    low_bound_com, high_bound_com = bootstrap_com_msd(coords, len(lag), pnum, args['bootstrap_trials'], spherical=not flat)
    if args['flat']:
        msd_com, _ = mto_com_msd(coords,times, max_lag=arg['max_lagtime'], orig_num=orig_num, delta=delta, spherical=False)
    else:
        #for spherical surfaces the center of mass is identically at the origin so we rely on this bootstrapping method
        #for an estimate
        msd_com = (low_bound_com+high_bound_com)/2
    dmsd_com = np.max([msd_com-low_bound_com,high_bound_com-msd_com],axis=0)

    #removing lattice diffusion
    msd_nocom = msd_ens - msd_com
    dmsd_nocom = dmsd_ens + dmsd_com

    np.save("mto_msd_ensemble.npy",np.array(lag,msd_ens,dmsd_ens))
    np.save("mto_msd_com.npy",np.array(lag,msd_com,dmsd_com))
    np.save("mto_msd_nocom.npy",,np.array(lag,msd_nocom,dmsd_nocom))

    #fitting to find long-time diffusive constant
    if flat:
        msd_func = lambda t, D0: 4*D0*t
    else:
        shell_radius = np.linalg.norm(coords[0:10],axis=-1).mean()
        msd_func = lambda x, D0: sphere_msd(x, D0, shell_radius=shell_radius)


    #finding short-long transition


    #performing fit


    output = {
            'D_L_ens': None,
            'dD_L_ens': None,
            'D_com': None,
            'dD_com': None,
            'D_diff': None,
            'dD_diff': None,
            'D_0': None,
    }

    dump_json(output, 'diffusion.json')