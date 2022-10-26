# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Plots several useful quantities--T, mean nearest-neighbot distance, etc--against time for a lammps run.

@author: Jack Bond, Alex Yeh
"""

import glob, os
import numpy as np

from FileHandling import read_infile, read_dump, read_thermo, get_thermo_time

from numpy.random import default_rng
import matplotlib.pyplot as plt

from timeit import default_timer as timer

from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

def plot_thermo(header, data, path):
    for i, item in enumerate(header):
        if item.startswith('Time') or item.startswith('Step'):
            continue
        else:
            fig, ax = plt.subplots()
            ax.plot(data[:,0], data[:,i])
            ax.set_xlabel("[$\\tau$]", fontsize=12)
            ax.set_title(item)
            fig.savefig(path+item+".jpg", bbox_inches='tight')
            plt.close(fig)

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

mass = 2.53e-14
a_hc = 1.4
kT = 4.1124e-21
tau = np.sqrt(kT/(mass*(2*a_hc*1e-6)**2))
pnum = multiple.shape[1]
# dt = lammps_params['timestep']*tau # [s]

damp = lammps_params['damp']
#kappa = lammps_params['kappa_2a']/(2*a_hc)
#bpp = lammps_params['bpp']
dt = lammps_params['timestep']

times = ts*dt
a_eff = a_hc#int_a_eff(a_hc, bpp, kappa)

if lammps_params['rad'] is None:
    shell_radii = np.linalg.norm(multiple,axis=-1).mean(axis=-1)
    shell_radius = shell_radii.min() #get stand in value for rough approximation
    all_eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radii**2)
    eta_eff = eta_eff.max() #get stand in value for rough approximation
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

start = timer()
min_dist = []
for i, frame in enumerate(multiple):
    dists = squareform(pdist(frame))
    min_dist.append(pdist(frame).min())

max_dist = max(min_dist)
fig, ax = plt.subplots()
ax.plot(ts*dt, min_dist)
ax.hlines(a_eff/a_hc, 0, np.max(ts*dt), ls=':', color='k', label='2a')
ax.set_xlabel("[$\\tau$]", fontsize=12)
ax.set_ylabel("[2a]")
ax.set_ylim([0.995, max(1.01*a_eff/a_hc, max_dist)])
ax.legend()
ax.set_title(f"minimum distance between particles")
fig.savefig(path+"all_plot_min_dist.jpg", bbox_inches='tight')
plt.close(fig)

#%% plotting thermo properties
plot_thermo(thermo_header, thermo, path)
end = timer()
print(f"minimum interparticle distance and thermo plots {end - start:.2f}s")