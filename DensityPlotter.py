# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 5, 2022

uses MCBatchAnalyzer to produce plots of denisty and area fraction
for a batch of simulations, including theoretical predictions from
ForceBalanceTheory

@author: Jack Bond
"""

import MCBatchAnalyzer
from MCBatchAnalyzer import *

simDicts, paramDicts, seedFoldersList, = categorize_batch()

#print(get_average_computation_time(seedFoldersList[0])[0])

#experimental conditions/assumptions
eta_c = 0.85

Rs = np.array([d['radius'] for d in simDicts])
vs = np.array([p['vpp'] for p in paramDicts])
print(Rs,vs)

a = paramDicts[0]['particle_radius']*1e6 #microns
aeff = units.getAEff(paramDicts[0])*1e6 #microns

#setting up density visualization
fig, ax = plt.subplots()
ax.set_title(f"Density Profiles for Phase Coexistence(?) on Surfaces of Varying Curvatures with $\eta_c={eta_c}$")
ax.set_xlabel("Arclength [$\mu m$]")
ax.set_ylabel(r"Density [$\mu m^{-2}$]")

fig2, ax2 = plt.subplots()
ax2.set_title(f"Comparison to Theoretical Area Fraction for Two-Phase Shells with $\eta_c={eta_c}$")
ax2.set_xlabel("Arclength [$\mu m$]")
ax2.set_ylabel(r"$\eta_{eff}$")
ax2.set_ylim([0,1])

for i,seedFolders in enumerate(seedFoldersList):
	mids, rho, eta, rs, eta_th = compute_SI_density_profile(seedFolders,paramDicts[i],simDicts[i],label=f"R_{Rs[i]}")
	ax2.plot(rs, eta_th,ls='--')
	ax.scatter(mids, rho, label=f'Radius: {Rs[i]*2*a:.2f} [$\mu m$], VPP: {vs[i]:.2f} [V]')
	ax2.scatter(mids, eta, label=f'Radius: {Rs[i]*2*a:.2f} [$\mu m$], VPP: {vs[i]:.2f} [V]')

ax.legend()#bbox_to_anchor=(1.5,1))
ax2.legend()#bbox_to_anchor=(1.5,1))

fig.savefig("Density.jpg", bbox_inches='tight')
fig2.savefig("AreaFraction.jpg", bbox_inches='tight')
