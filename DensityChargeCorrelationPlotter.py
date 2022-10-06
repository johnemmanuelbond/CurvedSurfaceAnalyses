# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

uses MCBatchAnalyzer to correlate local density (via voronoi areas) to
topological charge (via voronoi vertices)

@author: Jack Bond
"""
import numpy as np
import os, sys, json

import matplotlib as mpl
import matplotlib.pyplot as plt

from UnitConversions import getAEff
#from OrderParameters import radialDistributionFunction
from OrderParameters import Vc, rho_voronoi, rho_voronoi_shell
from MCBatchAnalyzer import categorize_batch, sample_frames

fig, [ax,axshell] = plt.subplots(1,2)
ax.set_title("Single Particle")
ax.set_xlabel("Topological Charge")
ax.set_ylabel(r"$\eta_{eff}$")
ax.set_ylim([0,1])
ax.axhline(y=0.69)
ax.axhline(y=0.71)

axshell.set_title("First Coord Shell")
axshell.set_xlabel("Topological Charge")
axshell.set_ylabel(r"$\eta_{eff}$")
axshell.set_ylim([0,1])
axshell.axhline(y=0.69)
axshell.axhline(y=0.71)

single = len(sys.argv) == 1 or sys.argv[1] != 'batch'
batch = sys.argv[1]=='batch'

if single:

	simFolder = os.getcwd()
	config = json.load(open(simFolder + "/configFile.json",'r'))
	simarg = config['simargument']
	params = config['params']
	N = simarg['npart']
	R = simarg['radius']
	a = params["particle_radius"]
	aeff = getAEff(params)
	eta_eff = N*(aeff/(2*a))**2/(4*R**2)

	lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
	pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

	frames = np.array(sample_frames([simFolder+"/"],label=lab,reset=True))

	qs = np.array([6-Vc(frame, R = R) for frame in frames]).flatten()
	#XS = 0.5*(np.array([np.sum(np.abs(q)) for q in qs])/12-1)
	etas = np.array([rho_voronoi(frame,R=R) for frame in frames]).flatten()*np.pi*(aeff/(2*a))**2
	etashells = np.array([rho_voronoi_shell(frame,R=R) for frame in frames]).flatten()*np.pi*(aeff/(2*a))**2

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
	ax.errorbar(pltqs,pltetashellss,yerr=pltdetashells,label=pltlab, ls='none', marker='^',fillstyle='none')

elif batch:
	configs, seedFoldersList = categorize_batch()

	Rs = np.array([c['simargument']['radius'] for c in configs])
	idx = np.argsort(Rs)	


	for i in idx:
		seedFolders = seedFoldersList[i]
		simarg = configs[i]['simargument']
		params = configs[i]['params']
		N = simarg['npart']
		R = simarg['radius']
		a = params["particle_radius"]
		aeff = getAEff(params)
		eta_eff = N*(aeff/(2*a))**2/(4*R**2)

		lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
		pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

		frames = np.array(sample_frames(seedFolders,label=lab,reset=True))

		qs = np.array([6-Vc(frame, R = R) for frame in frames]).flatten()
		#XS = 0.5*(np.array([np.sum(np.abs(q)) for q in qs])/12-1)
		etas = np.array([rho_voronoi(frame,R=R) for frame in frames]).flatten()*np.pi*(aeff/(2*a))**2
		etashells = np.array([rho_voronoi_shell(frame,R=R) for frame in frames]).flatten()*np.pi*(aeff/(2*a))**2

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
		ax.errorbar(pltqs,pltetashellss,yerr=pltdetashells,label=pltlab, ls='none', marker='^',fillstyle='none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig("DensityChargeCorrelation.jpg",bbox_inches='tight')

