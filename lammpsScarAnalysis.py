# -*- coding: utf-8 -*-
"""
Created on thu, Sep 29, 2022

Analyzes a lammps run for clusters of charge, as a well as the spatial coordination of grains scars.

@author: Jack Bond
"""

import glob, os
import numpy as np
from timeit import default_timer as timer

import UnitConversions as units
import FileHandling as handle
from MCBatchAnalyzer import scar_correlation


#set up plots
r_ico = np.sin(2*np.pi/5)
theta1 = 2*np.arcsin(1/2/r_ico)
theta2 = 2*np.arccos((r_ico**2+r_ico**2*np.cos(theta1/2)**2-3/4)/(2*r_ico**2*np.cos(theta1/2)))

figScar,axScar = plt.subplots()
axScar.set_xlabel(r"Geodesic Distance $[rad/$\pi]$")
axScar.set_ylim([0,2])
axScar.set_xlim([0,1])
axScar.set_ylabel(r"$g_{{scar-scar}}$")
axScar.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--',")
axScar.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')

figHist,[ax1,ax2,ax3] = plt.subplots(0,3)
ax1.set_ylabel("Counts")
ax1.set_xlabel("Cluster Size")
ax2.set_xlabel("Cluster Net Charge")
ax3.set_xlabel("Cluster Total Charge")

# load data
start = timer()
path = os.getcwd()+"/"
print(path)
#%% get run info
infile = glob.glob(path+'*.in')
assert len(infile) == 1, "need to have one specified input file"

a_hc = 1.4

lammps_params = handle.read_infile(infile[0])
time_str = handle.get_thermo_time(path+'log.lammps')
multiple = np.load(path+'datapts.npy')
ts = np.load(path+'times.npy')
fnum = multiple.shape[0]
pnum = multiple.shape[1]
coordination = np.load(path+'vor_coord.npy')
# dt = lammps_params['timestep']*tau # [s]

damp = lammps_params['damp']
kappa = lammps_params['kappa_2a']/(2*a_hc)
bpp = lammps_params['bpp']
dt = lammps_params['timestep']
R = lammps_params['rad']

times = ts*dt


midssScar =  []
gsScar = []

scarSizes = []
scarNetCharges = []
scarTotalCharges = []

#loop through equilibrated frames to gather information about charged clusters. We don't want to excessively calculate cluster positions, and we only want to sample independent frames, so I've created an array of index positions to control which frames we actually perform calculations on.
last_section = 1/3
desired_samples = 50
idx = np.arange(int((1-last_section)*fnum),fnum,(last_section*fnum)/samples)

for i in idx:
	frame = multiple[i]
	q = 6-coordination[i]
	midsScar, gScar, scars, meanscarpositions = scar_correlation(frame,R,bin_width=R*np.pi/40)
    midssScar.append(midsScar/(np.pi*R))
    gsScar.append(gScar)
    [scarSizes.append(s.size) for s in scars]
    [scarNetCharges.append(np.sum(q[s])) for s in scars]
    [scarTotalCharges.append(np.sum(np.abs(q[s]))) for s in scars]

midsScar = np.mean(np.array(midssScar),axis=0)
gScar = np.mean(np.array(gsScar),axis=0)
axScar.plot(midsScar, gScar,lw = 0.5)
figScar.savefig("Scar-Scar Pair Correlation.jpg", bbox_inches='tight')

scarSizes = np.array(scarSizes)
scarNetCharges = np.array(scarNetCharges)
scarTotalCharges = np.array(scarTotalCharges)

ax1.hist(scarSizes)
ax2.hist(scarNetCharges)
ax3.hist(scarTotalCharges)
figHist.savefig("Cluster Histrograms.jpg")