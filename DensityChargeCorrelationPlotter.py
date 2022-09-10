# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

uses MCBatchAnalyzer to correlate local density (via voronoi areas) to
topological charge (via voronoi vertices)

@author: Jack Bond
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from FileHandling import read_xyz_frame
#from OrderParameters import radialDistributionFunction
from OrderParameters import Vc, rho_voronoi
from MCBatchAnalyzer import categorize_batch, sample_frames

configs, seedFoldersList = categorize_batch()

fig, ax = plt.subplots()
ax.set_title("")
ax.set_xlabel("Topological Charge")
ax.set_ylabel(r"Local Density [$(2a)^{-2}$]")

for i,seedFolders in enumerate(seedFoldersList):
	simarg = configs[i]['simargument']
	params = configs[i]['params']
	N = simarg['npart']
	R = simarg['radius']
	a = params["particle_radius"]
	aeff = units.getAEff(params)
	eta_eff = N*(aeff/(2*a))**2/(4*R**2)

	lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
	pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

	frames = np.array(sample_frames(seedFolders,label=lab,reset=True))

	qs = [6-Vc(frame, R = R) for frame in frames].flatten()
	#XS = 0.5*(np.array([np.sum(np.abs(q)) for q in qs])/12-1)
	rhos = [rho_voronoi(frame,R=R) for frame in frames].flatten()

	pltqs=[]
	pltrhos=[]
	pltdrhos=[]

	for q in np.unique(qs):
		rho = rhos[qs==q].mean()
		drho = rhos[qs==q].std()

		pltqs.append(q)
		pltrhos.append(rho)
		pltdrhos.append(pltdrhos)

	ax.errorbar(pltqs,pltrhos,yerr=pltdrhos,label=pltlab)

ax.legend(loc = 5)

