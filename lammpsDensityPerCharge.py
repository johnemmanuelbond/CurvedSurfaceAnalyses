# -*- coding: utf-8 -*-
"""
Created on Sat, Oct 22, 2022

uses Alex Yeh's lammps analyss tools  to correlate local density (via voronoi areas) to
topological charge (via voronoi vertices)

@author: Jack Bond
"""
import numpy as np
from scipy import integrate
import os, sys, json, glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from timeit import default_timer as timer


import main_lib.FileHandling as handle
#from main_lib.OrderParameters import radialDistributionFunction
from main_lib.OrderParameters import Vc, rho_voronoi, rho_voronoi_shell


def int_a_eff(radius, Bpp, kappa):
    integrand = lambda r: 1-np.exp(-1*Bpp*np.exp(-1*kappa*r))
    
    debye_points = np.arange(5)/(kappa)
        
    first, fErr = integrate.quad(integrand, 0, 1000/kappa, points=debye_points)
    second, sErr = integrate.quad(integrand, 1000/kappa, np.inf)
        
    return radius + 1/2*(first+second)

fig, [ax,axshell] = plt.subplots(1,2,sharex=True,sharey=True)
ax.set_title("Single Particle")
ax.set_xlabel("Topological Charge")
ax.set_ylabel(r"$\eta_{eff}$")
ax.set_ylim([0,1])
ax.axhline(y=0.69)
ax.axhline(y=0.71)

axshell.set_title("First Coord Shell")
#axshell.set_xlabel("Topological Charge")
#axshell.set_ylabel(r"$\eta_{eff}$")
axshell.set_ylim([0,1])
axshell.axhline(y=0.69)
axshell.axhline(y=0.71)

single = len(sys.argv) == 1 or sys.argv[1] != 'batch'
if len(sys.argv)>=2: batch = sys.argv[1]=='batch'

if single:
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
	N = multiple.shape[1]
	coordination = np.load(path+'vor_coord.npy')
	# dt = lammps_params['timestep']*tau # [s]

	damp = lammps_params['damp']
	kappa = lammps_params['kappa_2a']/(2*a_hc)
	bpp = lammps_params['bpp']
	dt = lammps_params['timestep']
	R = lammps_params['rad']

	aeff = int_a_eff(a_hc, bpp, kappa)
	eta_eff = N*(aeff/(2*a_hc))**2/(4*R**2)

	#loop through equilibrated frames to gather information about charged clusters. We don't want to excessively calculate cluster positions, and we only want to sample independent frames, so I've created an array of index positions to control which frames we actually perform calculations on.
	last_section = 1/3
	desired_samples = 100
	idx = np.arange(int((1-last_section)*fnum),fnum,int((last_section*fnum)/desired_samples))

	lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
	pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

	frames = multiple[idx]

	qs = np.array([6-Vc(frame, R = R,tol=1e-5) for frame in frames]).flatten()
	#XS = 0.5*(np.array([np.sum(np.abs(q)) for q in qs])/12-1)
	etas = np.array([rho_voronoi(frame,R=R,tol=1e-5) for frame in frames]).flatten()*np.pi*(aeff/(2*a_hc))**2
	etashells = np.array([rho_voronoi_shell(frame,R=R,tol=1e-5) for frame in frames]).flatten()*np.pi*(aeff/(2*a_hc))**2

	pltqs=[]
	pltetas=[]
	pltetashells = []
	pltdetas=[]
	pltdetashells = []

	for q in np.unique(qs):
		eta = etas[qs==q].mean()
		etashell = etashells[qs==q].mean()
		counts = etas[qs==q].size
		print(f"q:{q}, counts:{counts}")
		deta = etas[qs==q].std()
		detashell = etashells[qs==q].std()

		pltqs.append(q)
		pltetas.append(eta)
		pltetashells.append(etashell)
		pltdetas.append(deta)
		pltdetashells.append(detashell)

	ax.errorbar(pltqs,pltetas,yerr=pltdetas,label=pltlab, ls='none', marker='^',fillstyle='none')
	axshell.errorbar(pltqs,pltetashells,yerr=pltdetashells,label=pltlab, ls='none', marker='^',fillstyle='none')

	handle.dumpDictionaryJSON({'qs':pltqs,'etas':pltetas,'detas':pltdetas,'etashells':pltetashells,'detashells':pltdetashells,'eta_eff':eta_eff,'R':R,'N':N},"ChargeAndDensity")

elif batch:
	outs = glob.glob("./*radius*etaeff*pnum*trial*/ChargeAndDensity.json")


	for o in outs:

		dic = json.load(open(o,'r'))
		eta_eff, R, N = dic['eta_eff'], dic['R'], dic['N']

		if eta_eff>0.68:

			lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
			pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

			pltqs= dic['qs']
			pltetas= dic['etas']
			pltetashells = dic['etashells']
			pltdetas = dic['detas']
			pltdetashells = dic['detashells']

			ax.errorbar(pltqs,pltetas,yerr=pltdetas,label=pltlab, ls='none', marker='^',fillstyle='none')
			axshell.errorbar(pltqs,pltetashells,yerr=pltdetashells,label=pltlab, ls='none', marker='^',fillstyle='none')

axshell.legend(loc='center right', bbox_to_anchor=(2.2, 0.5))

fig.savefig("DensityChargeCorrelation.jpg",bbox_inches='tight')

