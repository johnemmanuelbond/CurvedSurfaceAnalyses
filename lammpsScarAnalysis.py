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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

figHist,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.set_ylabel("Counts")
ax1.set_xlabel("Cluster Size")
ax2.set_xlabel("Cluster Net Charge")
ax3.set_xlabel("Cluster Total Charge")

fig3D = plt.figure()
ax3D = fig3D.add_subplot(projection='3d')
ax3D.set_xlabel("Cluster Net Charge")
ax3D.set_ylabel("Cluster Total Charge")

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
desired_samples = 100
idx = np.arange(int((1-last_section)*fnum),fnum,int((last_section*fnum)/desired_samples))


for i in idx:
	frame = multiple[i]
	R = np.linalg.norm(frame,axis=-1).mean()
	q = 6-coordination[i]
	midsScar, gScar, scars, meanscarpositions = scar_correlation(frame,R,bin_width=R*np.pi/40,tol=1e-5)
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

def bins(arr):
	start = min(arr) - 0.5
	end = max(arr) + 0.5
	return np.linspace(start,end,int(end-start+1))

ax1.hist(scarSizes,bins=bins(scarSizes))
ax2.hist(scarNetCharges,bins=bins(scarNetCharges))
ax3.hist(scarTotalCharges,bins=bins(scarTotalCharges))
figHist.savefig("Cluster Histrograms.jpg")

hist, xedges, yedges = np.histogram2d(scarNetCharges,scarTotalCharges,bins=[bins(scarNetCharges),bins(scarTotalCharges)])

xmids = (xedges[1:]+xedges[:-1])/2
ymids = (yedges[1:]+yedges[:-1])/2

X,Y = np.meshgrid(xmids,ymids,indexing="ij")

print(hist.shape,X.shape)

w=0.5

ax3D.bar3d(X.ravel()-w/2,Y.ravel()-w/2,0,w,w,hist.ravel())
ax3D.set_aspect("equal", "box")

fig3D.savefig("3DHistogram.jpg",bbox_inches='tight')