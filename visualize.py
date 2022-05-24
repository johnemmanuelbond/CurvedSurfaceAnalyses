# -*- coding: utf-8 -*-
"""
Created on Mon, Feb 28, 2022

Preps and opens files for viewing in ovito with colors according to
a specific order parameter.

@author: Jack Bond
"""

import numpy as np
import scipy as sp

import os, glob, sys, json

import OrderParameters as order
import UnitConversions as units

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from timeit import default_timer as timer

ovito = "\"C:\\Program Files\\OVITO Basic\\ovito.exe\""

def read_xyz_frame(filename):
	"""Reads xyz coordinates from a .xyz file. Expected format:
	number of particles
	comment line (should have units)
	type x-coord y-coord z-coord
	type x-coord y-coord z-coord
	
	Returns a triply nested numpy array, with format:
	[ # first frame
	 [-10.34, -10.8, 37.1], #coordinates of particle
	 [-14.48, 5.69, 36.85],
	 [-7.47, -16.2, 35.8],
	"""
	frame = []
	with open(filename, 'r') as xyzfile:
		pnum = None
		for i, line in enumerate(xyzfile):
			if i == 0:
				pnum = float(line.strip())
			elif i == 1:
				pass  #throw away comment
			elif i <= pnum+1:
				# assumes format as below for particle coordinates
				# index xcomp ycomp zcomp ... (unspecified afterwards)
				coordinates = line.split()[1:4]
				frame.append([float(coord) for coord in coordinates])
			else:
				print("extra: " + line)
	return np.array(frame)

"""
returns a good-lookin RGB array for an order parameter N, normalized to one for
6-fold order
"""
def getRGB(N):
	#gray particles
	r,g,b = 0.6, 0.6, 0.6

	#blue if it's on the border:
	if N == -1:
		r,g,b = 0.02, 0.02, 0.6
	# green if it's close to 7-fold
	elif N > 6:
		r,g,b = 0, 0.5*(N/6), 0.02
	#red if it's close to 5-fold
	elif N < 6:
		r,g,b = 1-0.5*(N/6),0, 0.02

	return r, g, b

def atomMovie(simFolder):
	simArgument = json.load(open(simFolder + "/simArgument.json",'r'))
	npart = simArgument['npart']
	R = simArgument['radius']
	nsnapfreq = simArgument['nsnap']
	
	mFile = open(f"{simFolder}/movie_voronoi.atom",'w')
	outputs = glob.glob(f"{simFolder}/output_*.xyz")
	nframes = len(outputs)

	for j in range(nframes-1):

		frame = read_xyz_frame(f"{simFolder}/output_{j+1}.xyz")
		voronoiNumber = order.Vc(frame,excludeborder=False,R=R)
		coordinationNumber = order.Nc(frame,shellradius=2.727566270839027)

		orderParameter = voronoiNumber
		#orderParameter = coordinationNumber

		relevant = orderParameter>0
		#relevant = orderParameter!=6

		frame = frame[relevant]
		orderParameter = orderParameter[relevant]


		mFile.write(f"ITEM: TIMESTEP\n{(j+1)*nsnapfreq}\n")
		mFile.write(f"ITEM: NUMBER OF ATOMS\n{frame.shape[0]+1}\n")
		mFile.write(f"ITEM: BOX BOUNDS ff ff ff\n{-R:e} {R:e}\n{-R:e} {R:e}\n{-R:e} {R:e}\n")
		mFile.write(f"ITEM: ATOMS id type xs ys zs Color Color Color\n")
		
		mFile.write(f"-1 {R-0.5} 0.5 0.5 0.5 0.8, 0.8, 0.9\n")
		for i, N in enumerate(orderParameter):
			mFile.write(f"{i} 0.5 ")
			for c in frame[i]:
				mFile.write(f"{c/(2*R)+0.5} ")
			r,g,b = getRGB(N)
			mFile.write(f"{r} {g} {b}\n")

	mFile.close()

from MCBatchAnalyzer import sample_frames, firstCoordinationShell, pair_charge_correlation, scar_correlation, BoltzmannInversion

if __name__=="__main__":
	framepath = ""

	r_ico = np.sin(2*np.pi/5)
	theta1 = 2*np.arcsin(1/2/r_ico)
	theta2 = 2*np.arccos((r_ico**2+r_ico**2*np.cos(theta1/2)**2-3/4)/(2*r_ico**2*np.cos(theta1/2)))

	fig55,ax55 = plt.subplots()
	ax55.set_xlabel(r"Geodesic Distance [rad/$\pi]$")
	ax55.set_ylim([0,2])
	ax55.set_xlim([0,1])
	ax55.set_ylabel(r"$g_{{5-5}}$")
	ax55.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
	ax55.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')
	
	figScar,axScar = plt.subplots()
	axScar.set_xlabel(r"Geodesic Distance [rad/$\pi]$")
	axScar.set_ylim([0,2])
	axScar.set_xlim([0,1])
	axScar.set_ylabel(r"$g_{{scar-scar}}$")
	axScar.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--',")
	axScar.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')

	figCharge, axCharge = plt.subplots()
	axCharge.set_title("Excess Charge vs Sweeps")
	axCharge.set_ylabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axCharge.set_xlabel("sweeps")

	nargs = len(sys.argv)
	if nargs <= 1:
		simFolder = os.getcwd()
		atomMovie(simFolder)

		simArgument = json.load(open(simFolder+"/simArgument.json",'r'))
		params = json.load(open(simFolder+"/params.json",'r'))
		a = params['particle_radius']
		aeff = units.getAEff(params)
		R = simArgument['radius']
		N = simArgument['npart']

		eta_eff = np.round(N*(aeff/(2*a))**2/(4*R**2),3)

		frames = sample_frames([simFolder+"/"],label="movie_voronoi",last_section=1/2,reset=True)
		initFrame = read_xyz_frame("output_0.xyz")
		_, info = order.radialDistributionFunction(initFrame)
		spacing = info['particle_spacing']
		print(spacing)

		ax55.set_title(rf"$\eta_{{eff}}$ = {eta_eff}, R/a = {R/spacing:.2f}")
		axScar.set_title(rf"$\eta_{{eff}}$ = {eta_eff}, R/a = {R/spacing:.2f}")

		midss55 = []
		gs55 = []
		midssScar = []
		gsScar = []

		for frame in frames:
			#R = np.mean(np.linalg.norm(frame,axis=-1))

			mids55, g55, _, _ = pair_charge_correlation(1,1,frame,R,bin_width=R*np.pi/40)
			midss55.append(mids55/(np.pi*R))
			gs55.append(g55)

			midsScar, gScar, _, _ = scar_correlation(frame,R,bin_width=R*np.pi/40)
			midssScar.append(midsScar/(np.pi*R))
			gsScar.append(gScar)

		mids55 = np.mean(np.array(midss55),axis=0)
		g55 = np.mean(np.array(gs55),axis=0)
		midsScar = np.mean(np.array(midssScar),axis=0)
		gScar = np.mean(np.array(gsScar),axis=0)


		ax55.plot(mids55, g55,lw = 0.5)
		#plt.show()
		fig55.savefig("5-5 Pair Correlation.jpg", bbox_inches='tight')

		axScar.plot(midsScar, gScar,lw = 0.5)
		#plt.show()
		figScar.savefig("Scar-Scar Pair Correlation.jpg", bbox_inches='tight')

		sim = sample_frames([simFolder+"/"],label="movie_voronoi",last_section=1.0,reset=True)
			
		sweeps = simArgument['nsnap']*(1+np.arange(sim.shape[0]))
		vcs = [order.Vc(frame, R = R) for frame in sim]

		excessVC = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)
		axCharge.plot(sweeps,excessVC,lw=0.6)

		figCharge.savefig("Excess Charge per Sweep")

		hbin_edge = np.histogram_bin_edges(excessVC,bins=int(12*max(excessVC)+1),range=(-1/24,max(excessVC)+1/24))
		hval, hbin = np.histogram(excessVC[simArgument['nsweeps']//(3*simArgument['nsnap']):], bins = hbin_edge)
		widths = hbin_edge[1:] - hbin_edge[:-1]
		mids = hbin_edge[:-1] + widths/2

		fig, [axhist,axinvert] = plt.subplots(1,2)
		axhist.set_title("histogram of excess charge")
		axinvert.set_title("Boltzmann Inversion")
		axhist.set_xlabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
		axinvert.set_xlabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
		axhist.set_ylabel("counts")
		axinvert.set_ylabel("U/kT")
		axhist.bar(mids,hval,width = widths)
		mids, Ukt = BoltzmannInversion(mids, widths, hval)
		axinvert.plot(mids, Ukt)

		fig.savefig("Excess Charge Histogram")

	elif nargs == 2:
		framepath = sys.argv[1]
		frame = np.array(read_xyz_frame(framepath))

		R = np.mean(np.linalg.norm(frame,axis=-1))

		#coordinationNumber = order.Nc(frame,shellradius = order.firstCoordinationShell(frame))
		voronoiNumber = order.Vc(frame,excludeborder=False)

		print(order.firstCoordinationShell(frame))

		#diff = ((6-coordinationNumber)-(6-voronoiNumber))
		#uninteresting = (coordinationNumber==6)*(voronoiNumber == 6) 

		orderParameter = voronoiNumber#+5*diff

		relevant = orderParameter>0#frame[:,2]>0

		charge = 6-orderParameter

		excessCharge = 0.5*(np.sum(np.abs(charge))/12-1)
		print(f"Total Charge {np.sum(charge)}\nExcess Charge: {excessCharge}")

		start = timer()
		scars, scarCharges = order.findScars(frame)
		end = timer()
		print(f"{end-start}s scarfinding time")
		scarindex = np.zeros((len(orderParameter),4))
		for i,scar in enumerate(scars):
			scarindex[scar] = cm.hsv(i/len(scars))
			# if(scarCharges[i]==0):
			# 	scarindex[scar] = np.array([2,0,0,0])

		vFile = open("visual.xyz", "w")
		vFile.write(f"{orderParameter[relevant].size+1}\n")
		vFile.write(f"{orderParameter.size} total particles snapshot\n")
		vFile.write(f"{R-0.5} 0.0 0.0 0.0 0.8, 0.8, 0.9\n")
		for i, N in enumerate(orderParameter[relevant]):
			vFile.write("0.5 ")
			for c in frame[relevant][i]:
				vFile.write(f"{c} ")
			r,g,b,_ = scarindex[i]#getRGB(N)
			vFile.write(f"{r} {g} {b}\n")

		vFile.close()
		os.system(f"{ovito} visual.xyz")

		mids55, g55, _, _ = pair_charge_correlation(1,1,frame,R,bin_width=R*np.pi/20)
		mids55*=1/(np.pi*R)

		ax55.plot(mids55, g55,lw = 0.5)
		#plt.show()
		fig55.savefig("5-5 Pair Correlation.jpg", bbox_inches='tight')

		midsScar, gScar, _, _ = scar_correlation(frame,R,bin_width=R*np.pi/20)
		midsScar*=1/(np.pi*R)

		axScar.plot(midsScar, gScar,lw = 0.5)
		#plt.show()
		figScar.savefig("Scar-Scar Pair Correlation.jpg", bbox_inches='tight')


	elif nargs > 2:
		raise Exception("You entered too much stuff, fool")

	