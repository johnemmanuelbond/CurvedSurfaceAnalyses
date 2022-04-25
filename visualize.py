# -*- coding: utf-8 -*-
"""
Created on Mon, Feb 28, 2022

Preps and opens files for viewing in ovito with colors according to
a specific order parameter.

@author: Jack Bond
"""

import numpy as np
import scipy as sp
from scipy.spatial import SphericalVoronoi, geometric_slerp

import os, glob, sys, json
import OrderParameters as order

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
	
	mFile = open(f"{simFolder}/movie.atom",'w')
	outputs = glob.glob(f"{simFolder}/output_*.xyz")
	nframes = len(outputs)

	for j in range(nframes-1):
		mFile.write(f"ITEM: TIMESTEP\n{(j+1)*nsnapfreq}\n")
		mFile.write(f"ITEM: NUMBER OF ATOMS\n{npart}\n")
		mFile.write(f"ITEM: BOX BOUNDS ff ff ff\n{-1/2:e} {1/2:e}\n{-1/2:e} {1/2:e}\n{-1/2:e} {1/2:e}\n")
		mFile.write(f"ITEM: ATOMS id type xs ys zs Color Color Color\n")

		frame = read_xyz_frame(f"{simFolder}/output_{j+1}.xyz")
		voronoiNumber = order.Vc(frame,excludeborder=True,R=R)
		
		for i, N in enumerate(voronoiNumber):
			mFile.write(f"{i} C ")
			for c in frame[i]:
				mFile.write(f"{c} ")
			r,g,b = getRGB(N)
			mFile.write(f"{r} {g} {b}\n")

	mFile.close()

if __name__=="__main__":
	framepath = ""

	nargs = len(sys.argv)
	if nargs <= 1:
		simFolder = os.getcwd()
		atomMovie(simFolder)
	elif nargs == 2:
		framepath = sys.argv[1]
		frame = np.array(read_xyz_frame(framepath))
		coordinationNumber = order.Nc(frame)
		voronoiNumber = order.Vc(frame)
		charge = 6-voronoiNumber
		charge[charge==7] = 0

		print(charge, np.sum(charge))

		vFile = open("visual.xyz", "w")
		vFile.write(f"{voronoiNumber.size}\n")
		vFile.write("snapshot\n")
		for i, N in enumerate(voronoiNumber):
			vFile.write("C ")
			for c in frame[i]:
				vFile.write(f"{c} ")
			r,g,b = getRGB(N)
			vFile.write(f"{r} {g} {b}\n")
		vFile.close()
		os.system(f"{ovito} visual.xyz")

	elif nargs > 2:
		raise Exception("You entered too much stuff, fool")

	