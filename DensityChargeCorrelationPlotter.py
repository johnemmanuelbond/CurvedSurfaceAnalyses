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
from OrderParameters import Vc, rho_voronoi
from MCBatchAnalyzer import categorize_batch, sample_frames

fig, ax = plt.subplots()
ax.set_title("")
ax.set_xlabel("Topological Charge")
ax.set_ylabel(r"Local Density [$(2a)^{-2}$]")

# single = len(sys.argv) == 1 or sys.argv[1] != 'batch'
# batch = sys.argv[1]=='batch'

# if single:

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
rhos = np.array([rho_voronoi(frame,R=R) for frame in frames]).flatten()

pltqs=[]
pltrhos=[]
pltdrhos=[]

for q in np.unique(qs):
	if(qs[qs==q].size > 10):
		rho = rhos[qs==q].mean()
		drho = rhos[qs==q].std()

		pltqs.append(q)
		pltrhos.append(rho)
		pltdrhos.append(pltdrhos)

ax.errorbar(pltqs,pltrhos,yerr=pltdrhos,label=pltlab)

# elif batch:
# 	configs, seedFoldersList = categorize_batch()

# 	for i,seedFolders in enumerate(seedFoldersList):
# 		simarg = configs[i]['simargument']
# 		params = configs[i]['params']
# 		N = simarg['npart']
# 		R = simarg['radius']
# 		a = params["particle_radius"]
# 		aeff = getAEff(params)
# 		eta_eff = N*(aeff/(2*a))**2/(4*R**2)

# 		lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
# 		pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

# 		frames = np.array(sample_frames(seedFolders,label=lab,reset=True))

# 		qs = np.array([6-Vc(frame, R = R) for frame in frames]).flatten()
# 		#XS = 0.5*(np.array([np.sum(np.abs(q)) for q in qs])/12-1)
# 		rhos = np.rray([rho_voronoi(frame,R=R) for frame in frames]).flatten()

# 		pltqs=[]
# 		pltrhos=[]
# 		pltdrhos=[]

# 		for q in np.unique(qs):
# 			rho = rhos[qs==q].mean()
# 			drho = rhos[qs==q].std()

# 			pltqs.append(q)
# 			pltrhos.append(rho)
# 			pltdrhos.append(pltdrhos)

# 		ax.errorbar(pltqs,pltrhos,yerr=pltdrhos,label=pltlab)

ax.legend(loc = 5)

fig.savefig("DensityChargeCorrelation.jpg")

