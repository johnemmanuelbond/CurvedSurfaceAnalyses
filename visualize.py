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

import os, glob, sys

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

#for planar systems only
def Ci6(coordinates, shellradius = 1.6):
	npart = coordinates.shape[0]
	i,j = np.mgrid[0:npart,0:npart]
	#dr = np.sqrt((coords[p,0]-coords[q,0])**2+(coords[p,1]-coords[q,1])**2)
	dr_vec = coordinates[i]-coordinates[j]
	dr_norm = np.linalg.norm(dr_vec,axis=-1)
	dr_norm[i==j] = 1e-3
	
	#to get the theta's we need an aribrtrary reference angle, we'll say (0,1) is our reference angle
	# we get the cosine with the dot product between dr and (0,1), and then divide it by the norm
	cosines = dr_vec[:,:,0]/dr_norm
	#fix nans:
	cosines[i==j]=0
	
	neighbors = dr_norm<shellradius
	neighbors[i==j]=False
	Nc = np.sum(neighbors,axis=-1)
	Nc[Nc==0]=1
	
	with np.errstate(divide='ignore'):

		argument = np.exp(6j * np.arccos(cosines))

		psi6 = np.array([(1/Nc[n])*np.sum(neighbors[n]*argument[n]) for n in range(npart)])

		psi6star = np.conjugate(psi6)

		chi6 = np.abs(np.real(psi6[i]*psi6star[j]))/np.abs(psi6[i]*psi6star[j])

		C6 = np.array([(1/6)*np.sum(neighbors[n]*(chi6[n]>0.32)) for n in range(npart)])

	return C6, Nc

def nearBoundary(coordinates, aFrac = 0.906):
	N = coordinates.shape[0]
	R = np.round(np.mean(np.linalg.norm(coordinates,axis=1)),5)
	zs = coordinates[:,2]

	maxZ = N/(8*aFrac*R)
	border = ((R-zs)-maxZ)
	border[border<-0.5]=0
	return border


#a simple coordination number 
def Nc(coordinates, shellradius = 1.6):
	npart = coordinates.shape[0]
	i,j = np.mgrid[0:npart,0:npart]
	dr_vec = coordinates[i]-coordinates[j]
	dr_norm = np.linalg.norm(dr_vec,axis=-1)
	dr_norm[i==j] = 0
	
	neighbors = dr_norm<shellradius
	neighbors[i==j]=False
	Nc = np.sum(neighbors,axis=-1)

	return Nc

#coordination number based of voronoi triangulation
def Vc(coordinates):
	radius = np.mean(np.linalg.norm(coordinates,axis=1))
	sv = SphericalVoronoi(coordinates, radius = radius)
	Vc = np.array([len(region) for region in sv.regions])

	border = nearBoundary(coordinates, aFrac = 0.9)
	#if you're on the other side of the arbitrary border you're colored blue
	Vc[border>0] = -1
	#if you're close to the border and you're 5-coordinated, you're colored blue
	Vc[(Vc<6)*(border<0)] = -1

	#border = 1*(sv.calculate_areas() > 1.2*np.pi/(4*0.71))

	return Vc

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

if __name__=="__main__":
	framepath = ""
	coordshell = (1.44635/1.4)*0.5*(1+np.sqrt(3))
	#print(coordshell)

	nargs = len(sys.argv)
	if nargs <= 1:
		raise Exception("No file Given")
	elif nargs == 2:
		framepath = sys.argv[1]
	elif nargs == 3:
		framepath = sys.argv[1]
		coordshell = float(sys.argv[2])
	elif nargs > 3:
		raise Exception("You entered too much stuff, fool")

	frame = np.array(read_xyz_frame(framepath))
	coordinationNumber = Nc(frame, shellradius=coordshell)
	voronoiNumber = Vc(frame)
	print(voronoiNumber)


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