# -*- coding: utf-8 -*-
"""
Created on thu, Sep 29, 2022

Turns a lammps run into a movie viewable in Ovito
Mostly copied from a version of Alex's 'visualizing.py' circa 8/25/22

@author: Alex Yeh, Jack Bond
"""

import glob, os
import numpy as np
from timeit import default_timer as timer

import UnitConversions as units
import FileHandling as handle

from FileHandling import read_infile, get_thermo_time

# load data
start = timer()
path = os.getcwd()+"/"
print(path)
#%% get run info
infile = glob.glob(path+'*.in')
assert len(infile) == 1, "need to have one specified input file"

lammps_params = handle.read_infile(infile[0])
time_str = handle.get_thermo_time(path+'log.lammps')
multiple = np.load(path+'datapts.npy')
ts = np.load(path+'times.npy')
pnum = multiple.shape[1]
coordination = np.load(path+'vor_coord.npy')
vor_col = np.array([coord_num_color(c) for c in coordination])
# dt = lammps_params['timestep']*tau # [s]

damp = lammps_params['damp']
kappa = lammps_params['kappa_2a']/(2*a_hc)
bpp = lammps_params['bpp']
dt = lammps_params['timestep']

times = ts*dt

shell_radius = lammps_params['rad']
a_eff = int_a_eff(a_hc, bpp, kappa)
eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radius**2)
title = (f"N: {pnum}, "
         +f"R: {shell_radius:0.3f} "
         +r"[2a], $\eta_{eff}$:" 
         + f"{eta_eff:0.3f}")

output_vis(path+"movie_voronoi.atom", 
           multiple, ts=ts,
           colors=vor_col)