# -*- coding: utf-8 -*-
"""
Created on Wed, Jun 6, 2022

uses MCBatchAnalyzer to produce plots of the boltzmann-inverted
energy in some coordinate

@author: Jack Bond
"""
import MCBatchAnalyzer
from MCBatchAnalyzer import *

simDicts, paramDicts, seedFoldersList = categorize_batch()

for i,seedFolders in enumerate(seedFoldersList):
	
	#label generation
	N = simDicts[i]['npart']
	R = simDicts[i]['radius']
	a = paramDicts[i]["particle_radius"]
	aeff = units.getAEff(paramDicts[i])
	eta_eff = N*(aeff/(2*a))**2/(4*R**2)

	initFrame = read_xyz_frame(seedFolders[0]+"output_0.xyz")
	_,info = order.radialDistributionFunction(initFrame)
	spacing = info['particle_spacing']

	lab = f"eta_eff={eta_eff:.3f},R={R:.1f}"
	Ra = R/spacing

	#reading data
	frames = np.array(sample_frames(seedFolders,label=lab,reset=True))

	#determining excess charge
	vcs = [order.Vc(frame, R = R) for frame in frames]
	XS = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)
	
	#Boltzmann inverting XS Charge
	hbin_edgeXS = np.histogram_bin_edges(XS,bins=int(12*max(XS)+1),range=(-1/24,max(XS)+1/24))
	hvalXS, hbinXS = np.histogram(XS[simDicts[i]['nsweeps']//(3*simDicts[i]['nsnap']):], bins = hbin_edgeXS)
	widthsXS = hbin_edgeXS[1:] - hbin_edgeXS[:-1]
	midsXS = hbin_edgeXS[:-1] + widthsXS/2
	
	figboltz, [axhist,axinvert] = plt.subplots(1,2)
	figboltz.suptitle(rf"N={N}, $\eta_{{eff}}$={eta_eff:.3f}, R/a={Ra:.1f}")
	axhist.set_title("Histogram of Excess Charge")
	axinvert.set_title("Boltzmann Inversion")
	axhist.set_xlabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axinvert.set_xlabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axhist.set_ylabel("counts")
	axinvert.set_ylabel("U/kT")
	axhist.bar(midsXS,hvalXS,width = widthsXS)
	midsInv, Ukt = BoltzmannInversion(midsXS, widthsXS, hvalXS)
	axinvert.plot(midsInv, Ukt)

	figboltz.savefig(f"Excess Charge Histogram - {lab}.jpg")