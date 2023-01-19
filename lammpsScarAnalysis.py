# -*- coding: utf-8 -*-
"""
Created on thu, Sep 29, 2022

Analyzes a lammps run for clusters of charge, as a well as the spatial coordination of grains scars.

@author: Jack Bond
"""

import glob, os, sys
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from main_lib import UnitConversions as units
from main_lib import FileHandling as handle
from main_lib.Correlation import scar_correlation, theta1, theta2

from timeit import default_timer as timer

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

from scipy import integrate
def int_a_eff(radius, Bpp, kappa):
	integrand = lambda r: 1-np.exp(-1*Bpp*np.exp(-1*kappa*r))

	debye_points = np.arange(5)/(kappa)

	first, fErr = integrate.quad(integrand, 0, 1000/kappa, points=debye_points)
	second, sErr = integrate.quad(integrand, 1000/kappa, np.inf)

	return radius + 1/2*(first+second)

aeff = int_a_eff(a_hc, bpp, kappa)
eta_eff = pnum*(aeff/(2*a_hc))**2/(4*R**2)
lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={pnum}"
pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={pnum}"

times = ts*dt


#Set up data structures

midssScar =  []
gsScar = []
#midssBackground =  []
#gsBackground = []

scarSizes = []
scarNetCharges = []
scarTotalCharges = []

#loop through equilibrated frames to gather information about charged clusters. We don't want to excessively calculate cluster positions, and we only want to sample independent frames, so I've created an array of index positions to control which frames we actually perform calculations on.
last_section = 1/3
desired_samples = 100
idx = np.arange(int((1-last_section)*fnum),fnum,int((last_section*fnum)/desired_samples))

figScar,axScar = plt.subplots()
axScar.set_xlabel(r"Geodesic Distance [rad/$\pi$]")
axScar.set_title(pltlab)
axScar.set_ylim([0,2])
axScar.set_xlim([0,1])
axScar.set_ylabel(r"$g_{{scar-scar}}$")
axScar.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--',")
axScar.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')

for i in idx:
	frame = multiple[i]
	R = np.linalg.norm(frame,axis=-1).mean()
	q = 6-coordination[i]
	midsScar, gScar, scars, meanscarpositions = scar_correlation(frame,R,bin_width=R*np.pi/40,tol=1e-5)
	#midsBackground, gBackground, _, _ = scar_correlation(frame,R,bin_width=R*np.pi/40,tol=1e-5,charge_to_correlate = 0)
	midssScar.append(midsScar/(np.pi*R))
	gsScar.append(gScar)
	#midssBackground.append(midsBackground/(np.pi*R))
	#gsBackground.append(gBackground)
	[scarSizes.append(s.size) for s in scars]
	[scarNetCharges.append(np.sum(q[s])) for s in scars]
	[scarTotalCharges.append(np.sum(np.abs(q[s]))) for s in scars]

midsScar = np.mean(np.array(midssScar),axis=0)
gScar = np.mean(np.array(gsScar),axis=0)
axScar.plot(midsScar, gScar,lw = 0.5,label = r"$\sum q = 1$")
#midsBackground = np.mean(np.array(midssBackground),axis=0)
#gBackground = np.mean(np.array(gsBackground),axis=0)
#axScar.plot(midsBackground, gBackground,lw = 0.5,label = r"$\sum q = 0$")
axScar.legend()
figScar.savefig("Scar-Scar Pair Correlation.jpg", bbox_inches='tight')

scarSizes = np.array(scarSizes)
scarNetCharges = np.array(scarNetCharges)
scarTotalCharges = np.array(scarTotalCharges)

#Histograms of each scar quality: net cahrge, total charge, size
figHist,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.set_ylabel("Counts")
ax1.set_xlabel("Cluster Size")
ax2.set_xlabel("Cluster Net Charge")
ax3.set_xlabel("Cluster Total Charge")
ax2.set_title(pltlab)

def bins(arr):
	start = min(arr) - 0.5
	end = max(arr) + 0.5
	return np.linspace(start,end,int(end-start+1))

ax1.hist(scarSizes,bins=bins(scarSizes))
ax2.hist(scarNetCharges,bins=bins(scarNetCharges))
ax3.hist(scarTotalCharges,bins=bins(scarTotalCharges))
figHist.savefig("Cluster Histrograms.jpg")

#complete 3D histogram of net charge and cluster size
fig3D = plt.figure()
ax3D = fig3D.add_subplot(projection='3d')
ax3D.set_xlabel("Cluster Net Charge")
ax3D.set_ylabel("Cluster Size")
ax3D.set_title(pltlab)

hist, xedges, yedges = np.histogram2d(scarNetCharges,scarSizes,bins=[bins(scarNetCharges),bins(scarTotalCharges)])

xmids = (xedges[1:]+xedges[:-1])/2
ymids = (yedges[1:]+yedges[:-1])/2

X,Y = np.meshgrid(xmids,ymids,indexing="ij")

#print(hist)

w=0.5

ax3D.bar3d(X.ravel()-w/2,Y.ravel()-w/2,0,w,w,hist.ravel())

fig3D.savefig("3DHistogram.jpg",bbox_inches='tight')


#Stacked histograms of cluster size per each net charge
figHists, axHists = plt.subplots()
axHists.set_xlabel("Cluster Size")
axHists.set_ylabel("Counts")
axHists.set_title(pltlab)

cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, x in enumerate(ymids):
	ser = hist[:,i]
	print(ser)
	js = np.flip(np.argsort(ser))
	for j in js:
		y = ser[j]
		q = xmids[j]
		c = cs[np.where(xmids==q)[0][0]]
		if i == 0:
			axHists.bar(x,y,2/3,label=rf"$\sum$q={q}",color=c)
		else: axHists.bar(x,y,2/3,color=c)

axHists.legend()
figHists.savefig("BarChart.jpg")

# 2D projection of the complete 3D histogram plus a boltzmann inversion
figgrid, axgrid = plt.subplots(figsize=(5,5))
axgrid.set_xlabel("Cluster Net Charge")
axgrid.set_ylabel("Cluster Size")
axgrid.set_title(pltlab)

X,Y = np.meshgrid(xedges,yedges,indexing='ij')
p = hist/(hist.sum().sum())
p[p==0] = np.min(p[p!=0].flatten())*np.exp(-5) #make sure high-energy clusters appear so on the grid.
e = -np.log(p) #kT
g = axgrid.pcolormesh(X,Y,e,cmap='coolwarm')
#axgrid.set_aspect('equal','box')
figgrid.colorbar(g,label='Energy [kT]',extend='max',spacing='uniform',ticks=np.arange(11))
figgrid.savefig("ClusterGrid.jpg")