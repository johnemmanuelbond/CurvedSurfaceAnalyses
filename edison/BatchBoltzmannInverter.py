# -*- coding: utf-8 -*-
"""
Created on Wed, Jun 6, 2022

uses MCBatchAnalyzer to produce plots of the boltzmann-inverted
energy in some coordinate

@author: Jack Bond
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
pwd = os.path.dirname(__file__)
sys.path.append(os.path.realpath(pwd+'/..'))
from FileHandling import read_xyz_frame
from OrderParameters import radialDistributionFunction
from OrderParameters import C6, Vc
from MCBatchAnalyzer import categorize_batch, sample_frames
from MCBatchAnalyzer import BoltzmannInversion

configs, seedFoldersList = categorize_batch()

Ns = []
XSs = []
delC6s = []
keys = []
ps = []

for i,seedFolders in enumerate(seedFoldersList):

	#label generation
	config = configs[i]#json.load(open(seedFolders[0]+"configFile.json"))
	key = config['interactions'][0]['key']
	mag = config['interactions'][0]['A']
	par = config['interactions'][0]['p']
	N = config['simargument']['npart']
	R = config['simargument']['radius']
	a = config['params']["particle_radius"]
	#aeff = units.getAEff(paramDicts[i])
	#eta_eff = N*(aeff/(2*a))**2/(4*R**2)

	initFrame = read_xyz_frame(seedFolders[0]+"output_0.xyz")
	_,info = radialDistributionFunction(initFrame)
	spacing = info['particle_spacing']

	lab = f"{N},{R:.3f}-{key};{mag:.3f},{par:.3f}"
	Ra = R/spacing

	Ns.append(N)
	ps.append(par)
	keys.append(key)

	C6s_0, meanC6_0,_,_ = C6(initFrame)

	#reading data
	frames = np.array(sample_frames(seedFolders,label=lab))

	#determining excess charge
	vcs = [Vc(frame, R = R) for frame in frames]
	XS = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)

	#Boltzmann inverting XS Charge
	hbin_edgeXS = np.histogram_bin_edges(XS,bins=int(12*max(XS)+1),range=(-1/24,max(XS)+1/24))
	hvalXS, hbinXS = np.histogram(XS, bins = hbin_edgeXS)
	widthsXS = hbin_edgeXS[1:] - hbin_edgeXS[:-1]
	midsXS = hbin_edgeXS[:-1] + widthsXS/2

	#plotting XS Charge histogram
	figboltz, [axhist,axinvert] = plt.subplots(1,2)
	figboltz.suptitle(rf"N={N}, {key}; {mag:.3f}, {par:.3f}")
	axhist.set_title("Histogram of Excess Charge")
	axinvert.set_title("Boltzmann Inversion")
	axhist.set_xlabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axinvert.set_xlabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axhist.set_ylabel("counts")
	axinvert.set_ylabel("U/kT")
	axhist.bar(midsXS,hvalXS,width = widthsXS)


	#boltzmann inversion and fitting result
	midsInv, Ukt = BoltzmannInversion(midsXS, widthsXS, hvalXS)
	from scipy.optimize import curve_fit as cf
	def para(x,A,x0,y0):
			return A*(x-x0)**2 + y0

	guess=np.array([200,1,0])
	fit, pcov = cf(para,midsInv[Ukt!=np.inf],Ukt[Ukt!=np.inf], p0=guess)
	Ufit= para(midsInv,*fit)

	axinvert.plot(midsInv, Ukt, label="data")
	axinvert.plot(midsInv, Ufit, label=rf"{fit[0]:.3f}(x-{fit[1]:.3f})$^2$ + {fit[2]:.3f}")
	axinvert.legend()

	figboltz.savefig(f"Excess Charge Histogram - {lab}.jpg")
	plt.close()

	XSs.append(fit[1])

	#determining average C6
	meanC6s = []
	for frame in frames:
		C6, meanC6, _, _ = C6(frame)
		meanC6s.append(meanC6)
	meanC6s = np.array(meanC6s)

	#Boltzmann inverting C6
	hbin_edgeC6 = np.histogram_bin_edges(meanC6s-meanC6_0,bins=20)
	hvalC6, hbinC6 = np.histogram(meanC6s-meanC6_0, bins = hbin_edgeC6)
	widthsC6 = hbin_edgeC6[1:] - hbin_edgeC6[:-1]
	midsC6 = hbin_edgeC6[:-1] + widthsC6/2

	#plotting C6 Charge histogram
	figboltz, [axhist,axinvert] = plt.subplots(1,2)
	figboltz.suptitle(rf"N={N}, {key}; {mag:.3f}, {par:.3f}")
	axhist.set_title("Histogram of Average C6 Differences")
	axinvert.set_title("Boltzmann Inversion")
	axhist.set_xlabel(r"$<C6>-<C6>_0$")
	axinvert.set_xlabel(r"$<C6>-<C6>_0$")
	axhist.set_ylabel("counts")
	axinvert.set_ylabel("U/kT")
	axhist.bar(midsC6,hvalC6,width = widthsC6)

	#boltzmann inversion and fitting result
	midsInv, Ukt = BoltzmannInversion(midsC6, widthsC6, hvalC6)
	def para(x,A,x0,y0):
			return A*(x-x0)**2 + y0

	guess=np.array([500,-0.1,-2.0])
	fit, pcov = cf(para,midsInv[Ukt!=np.inf],Ukt[Ukt!=np.inf], p0=guess)
	Ufit= para(midsInv,*fit)

	axinvert.plot(midsInv, Ukt, label="data")
	axinvert.plot(midsInv, Ufit, label=rf"{fit[0]:.3f}(x-{fit[1]:.3f})$^2$ + {fit[2]:.3f}")
	axinvert.legend()

	figboltz.savefig(f"Mean C6 Histogram - {lab}.jpg")
	plt.close()

	delC6s.append(fit[1])

Ns = np.array(Ns)
XSs = np.array(XSs)
delC6s = np.array(delC6s)
keys = np.array(keys)
ps = np.array(ps)

figtrends, axtrends = plt.subplots()
axtrends.set_title("Most Probable XS Charge vs System Size")
axtrends.set_ylabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
axtrends.set_xlabel("N")
for p in np.unique(ps):
	xs = Ns[ps == p]
	ind = np.argsort(xs)
	ys = XSs[ps == p]
	key = keys[ps==p][0]
	axtrends.plot(xs[ind],ys[ind], label = f"{key}, {p:.4f}")
axtrends.legend()
figtrends.savefig("XS Charge Trends")

figtrends, axtrends = plt.subplots()
axtrends.set_title("Most Probable C6 Difference vs System Size")
axtrends.set_ylabel(r"$<C6>-<C6>_0$")
axtrends.set_xlabel("N")
for p in np.unique(ps):
	xs = Ns[ps == p]
	ind = np.argsort(xs)
	ys = delC6s[ps == p]
	key = keys[ps==p][0]
	axtrends.plot(xs[ind],ys[ind], label = f"{key}, {p:.4f}")
axtrends.legend()
figtrends.savefig("C6 diff Trends")