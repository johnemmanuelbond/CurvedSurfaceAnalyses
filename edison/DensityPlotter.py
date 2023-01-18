# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 5, 2022

uses MCBatchAnalyzer to produce plots of denisty and area fraction
for a batch of simulations, including theoretical predictions from
ForceBalanceTheory, as well as charge distributions as a function of
arclength

@author: Jack Bond
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
pwd = os.path.dirname(__file__)
sys.path.append(os.path.realpath(pwd+'/..'))
from MCBatchAnalyzer import categorize_batch
from UnitConversions import getAEff

configs, seedFoldersList = categorize_batch()

#print(get_average_computation_time(seedFoldersList[0])[0])

#experimental conditions/assumptions
eta_c = 0.85



Rs = np.array([c['simargument']['radius'] for c in configs])
vs = np.array([c['params']['vpp'] for c in configs])
print(Rs,vs)

a = configs[0]['params']['particle_radius']*1e6 #microns
aeff = getAEff(configs[0]['params'])*1e6 #microns

#setting up density visualization
fig, ax = plt.subplots()
ax.set_title(f"Charge Density Profiles, Vpp = 2V, Random Starting States")
ax.set_xlabel("Arclength [$\mu m$]")
ax.set_ylabel(r"Average Topological Charge Density $[\mu m^{-2}]$")

fig2, ax2 = plt.subplots()
ax2.set_title(f"Comparison to Theoretical Area Fraction, Vpp = 2V, Random Starting States")
ax2.set_xlabel("Arclength [$\mu m$]")
ax2.set_ylabel(r"$\eta_{eff}$")
ax2.set_ylim([0,1])

for i,seedFolders in enumerate(seedFoldersList):
	lab = f"R{Rs[i]:.2f}_V{vs[i]:.2f}"
	if(configs[i]['simargument']['datapath']=='?'):
		lab+="_rand"
	mids, charge, avg_charge = compute_topological_charge_profile(seedFolders,configs[i],label=lab)
	mids, rho, eta, rs, eta_th = compute_SI_density_profile(seedFolders,configs[i],label=lab)
	ax2.plot(rs, eta_th,ls='--')
	ax.scatter(mids, charge, label=lab)
	ax2.scatter(mids, eta, label=lab)

ax.legend(bbox_to_anchor=(1.5,1))
ax2.legend()#bbox_to_anchor=(1.5,1))

fig.savefig("ChargeDensity - Random.png", bbox_inches='tight')
fig2.savefig("AreaFraction - Random.png", bbox_inches='tight')

