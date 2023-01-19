# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Plots several useful quantities--T, mean nearest-neighbot distance, etc--against time for a lammps run.

@author: Jack Bond, Alex Yeh
"""

import glob, os, sys
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

from main_lib.FileHandling import read_infile, read_dump, read_thermo, get_thermo_time
from main_lib.OrderParameters import Vc

from timeit import default_timer as timer

allstart = timer()

# load data
path = './'
print(path)
#%% read dump file
reset = False

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


if (not reset) and os.path.exists(path+'vor_coord.npy'):
    coordination = np.load(path+'vor_coord.npy')
    coordination = np.array(coordination, dtype=np.int16)
else:
    coordination = np.ones(multiple.shape[:-1])
    colors = np.ones((*multiple.shape[:-1], 4))
    for i, frame in enumerate(multiple):
        coordination[i] = Vc(frame,tol=1e-5)
    np.save(path+'vor_coord.npy', coordination)
    
end = timer()
interval = end - start
print(f"read and process files {interval:.2f}s")