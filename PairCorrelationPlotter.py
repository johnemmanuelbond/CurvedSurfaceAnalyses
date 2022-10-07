# -*- coding: utf-8 -*-
"""
Created on Wed, Jun 6, 2022

uses MCBatchAnalyzer to produce plots of pair-pair correlation functions
for various topological charges and other things like grain scar centers
of mass

@author: Jack Bond
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from FileHandling import read_xyz_frame
from UnitConversions import getAEff
from OrderParameters import radialDistributionFunction

from MCBatchAnalyzer import categorize_batch, sample_frames
from MCBatchAnalyzer import firstCoordinationShell
from MCBatchAnalyzer import compute_average_scar_correlation, compute_average_pair_charge_correlation

configs, seedFoldersList = categorize_batch()

#print(get_average_computation_time(seedFoldersList[0])[0])

# Charge Correlation Visualization for 5s and 7s at once
# fig, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(15,5))
# fig.suptitle("Pair Correlation Functions for Long-Range Repulsion")
# ax2.set_xlabel(r"Geodesic Distance [rad/$\pi]$")
# ax1.set_title(r"$g_{{55}}$")
# ax2.set_title(r"$g_{{57}}$")
# ax3.set_title(r"$g_{{77}}$")

# ax1.set_ylim([0,2])
# ax1.set_xlim([0,1])
# ax2.set_ylim([0,2])
# ax2.set_xlim([0,1])
# ax3.set_ylim([0,2])
# ax3.set_xlim([0,1])

#+1 +1 charge correlation visualization
fig55,ax55 = plt.subplots()
ax55.set_title("5-5 Pair Correlation Functions for Long-Range Repulsion")
ax55.set_xlabel(r"Geodesic Distance [rad/$\pi$]")
ax55.set_ylim([0,2])
ax55.set_xlim([0,1])

#Scar Scar correlation visualization
figScar, axScar = plt.subplots()
axScar.set_title("Scar-Scar Correlation Functions for Long-Range Repulsion")
axScar.set_xlabel(r"Geodesic Distance [rad/$\pi$]")
axScar.set_ylim([0,2])
axScar.set_xlim([0,1])

#icosohedral angles
r_ico = np.sin(2*np.pi/5)
theta1 = 2*np.arcsin(1/2/r_ico)
theta2 = 2*np.arccos((r_ico**2+r_ico**2*np.cos(theta1/2)**2-3/4)/(2*r_ico**2*np.cos(theta1/2)))

# ax1.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
# ax1.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')
# ax2.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
# ax2.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')
# ax3.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
# ax3.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')
ax55.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
ax55.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')
axScar.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
axScar.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')

for i,seedFolders in enumerate(seedFoldersList):
	simarg = configs[i]['simargument']
	params = configs[i]['params']
	inter = configs[i]['interactions'][0]
	N = simarg['npart']
	R = simarg['radius']
	a = params["particle_radius"]
	aeff = getAEff(params)
	eta_eff = N*(aeff/(2*a))**2/(4*R**2)

	initFrame = read_xyz_frame(seedFolders[0]+"output_0.xyz")
	_,info = radialDistributionFunction(initFrame)
	spacing = info['particle_spacing']

	lab = f"eta_eff={eta_eff:.3f},R={R:.1f}"
	midsRad, hval, coordinationShell, _ = firstCoordinationShell(seedFolders, label=lab)
	Ra = R/spacing
	
	pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R/a={Ra:.1f}"

	frames = np.array(sample_frames(seedFolders,label=lab,reset=True))

	#use to filter what shows up in the plot, for example we can do eta_eff>0.71
	cond = True;

	if(cond):
		mids55, g55, Qs = compute_average_pair_charge_correlation(1,1,seedFolders,configs[i],label=lab)
		# mids57, g57, Qs = compute_average_pair_charge_correlation(1,-1,seedFolders,simDicts[i],label=lab)
		# mids77, g77, Qs = compute_average_pair_charge_correlation(-1,-1,seedFolders,simDicts[i],label=lab)

		#print(f"{lab}: Total Charge per Frame: {Qs}")
		
		# ax1.plot(mids55/np.pi, g55, label=pltlab,lw = 0.5)
		# ax2.plot(mids57/np.pi, g57, label=pltlab,lw = 0.5)
		# ax3.plot(mids77/np.pi, g77, label=pltlab,lw = 0.5)
		ax55.plot(mids55/np.pi, g55, label=pltlab,lw = 0.5)

		midsScar, gScar, scarCount = compute_average_scar_correlation(seedFolders, configs[i], label=lab)
		axScar.plot(midsScar/np.pi,gScar,label=pltlab,lw=0.5)

#ax1.legend();ax2.legend();ax3.legend();
ax55.legend(bbox_to_anchor=(1.5,1))
axScar.legend(bbox_to_anchor=(1.5,1))

#fig.savefig("Pair Correlations.png", bbox_inches='tight')
fig55.savefig(f"5-5 Pair Correlations.png", bbox_inches='tight')
figScar.savefig(f"Scar-Scar Correlations.png", bbox_inches='tight')