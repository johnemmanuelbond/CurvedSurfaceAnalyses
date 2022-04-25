import UnitConversions as units
from visualize import getRGB
from visualize import read_xyz_frame

import numpy as np
import scipy as sp
from scipy.spatial import SphericalVoronoi, geometric_slerp

import os, sys, glob, json

import matplotlib as mpl
import matplotlib.pyplot as plt

from timeit import default_timer as timer

"""from a a spherical cap in 3D coordinates, returns a 2D projection
in polar coordinates where the radius is the arclength from the pole and
the angle is the azimuthal angle. Also returns said coordinates in a
cartesian representation, and the jacobian at each point for considering length
scales"""
def capPolarProjection(frame):
	Rs = np.linalg.norm(frame,axis=-1)
	l = Rs*np.arccos(frame[:,2]/Rs)
	phi = np.arctan(frame[:,1]/(frame[:,0]+0.000001))+np.pi*(frame[:,0]<0)
	x,y = l*np.cos(phi), l*np.sin(phi)
	jacobian = Rs*np.sin(l/Rs)
	return np.array([x,y]).T, np.array([l,phi]).T, jacobian

#a simple coordination number 
def Nc(coordinates, shellradius = (1.44635/1.4)*0.5*(1+np.sqrt(3))):
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
def Vc(frame,excludeborder=False,R=None):
	minZ = min(frame[:,2])
	
	if R == None:
		radius = np.mean(np.linalg.norm(frame,axis=1))
	else:
		radius = R
	sv = SphericalVoronoi(frame, radius = radius)
	Vc = np.zeros(frame.shape[0])
	for i, region in enumerate(sv.regions):
		Vc[i] = len(region)
		for v in sv.vertices[region]:
			if(v[2]<minZ):
				Vc[i] -= 1
				if(excludeborder):
					Vc[i]=-1
	return Vc

def charge(frame, excludeborder=False,R=None):
	return 6-Vc(frame,excludeborder=excludeborder,R=R)

def totalCharge(frame,R=None):
	vc = Vc(frame,excludeborder=True,R=R)
	charge = 6-vc[vc!=-1]
	return charge, np.sum(charge)

def averageCharge(frame,R=None):
	vc = Vc(frame,excludeborder=True,R=R)
	charge = 6-vc[vc!=-1]
	return charge, np.mean(charge)

def shells(pnum):
    """from particle number, calculate number of shells assuming hexagonal crystal"""
    # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
    return -1/2 + np.sqrt((pnum-1)/3 + 1/4)

def c6_hex(pnum):
    """returns C6 for a hexagonal cluster of the same size"""
    # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
    s = shells(pnum)
    return 6*(3*s**2 + s)/pnum

def findNeighbors(frame):
	N = frame.shape[0]
	i,j = np.mgrid[0:N,0:N]

	vc = Vc(frame)
	
	dr_vec = frame[i]-frame[j]
	dr_norm = np.linalg.norm(dr_vec,axis=-1)
	dr_norm[i==j] = np.inf

	neighbors = 0*i
	for k, v in enumerate(vc):
		nearest = np.argsort(dr_norm[k])[:int(vc[k])]
		neighbors[k,nearest] = 1

	return neighbors, vc


def Psi6(frame, reference = np.array([0,1,0])):
	N = frame.shape[0]
	i,j = np.mgrid[0:N,0:N]
	n, vc = findNeighbors(frame)


	# finding the tanget-plane components of the nearest-neighbor bonds, we'll
	# call these unit vectors the bond orientations
	dr_vec = frame[i]-frame[j]
	bond_orientations = []
	for k, r in enumerate(frame):
		normal = r/np.linalg.norm(r)
		drk = dr_vec[k]
		bonds = drk[n[k]==1]
		bonds = bonds - np.array([np.dot(b,normal)*normal for b in bonds])
		orientation = np.array([b/np.linalg.norm(b,axis=-1) for b in bonds])
		bond_orientations.append(orientation)


	# bond orientation depends on a global reference vector. On the surface of
	# a sphere we parallel transport said reference vector along a geodesic
	# to the particle who's bonds we're trying to measure
	Rs = np.linalg.norm(frame,axis=-1)
	polar = np.array(np.arccos(frame[:,2]/Rs))
	azimuth = np.array(np.arctan(frame[:,1]/(frame[:,0]+1e-20))+np.pi*(frame[:,0]<0))

	axes = np.array([-1*np.sin(azimuth),np.cos(azimuth),0*azimuth]).T
	
	iden = np.array([np.eye(3) for ax in axes])
	outer = np.array([np.outer(ax,ax) for ax in axes])
	cross = np.array([np.array([[0*ax[0],-1*ax[2],ax[1]],
								[ax[2],0*ax[0],-1*ax[0]],
								[-1*ax[1],ax[0],0*ax[0]]]) for ax in axes])

	rotationMatrices = np.einsum('k,kij->kij',np.cos(polar),iden)
	rotationMatrices += np.einsum('k,kij->kij',np.sin(polar),cross)
	rotationMatrices += np.einsum('k,kij->kij',(1-np.cos(polar)),outer)


	references = np.einsum('aij,j->ai',rotationMatrices,reference)
	
	LC = np.zeros((3,3,3))
	LC[0,1,2],LC[1,2,0],LC[2,0,1] = 1.0,1.0,1.0
	LC[2,1,0],LC[0,2,1],LC[1,0,2] = -1.0,-1.0,-1.0
	#print(LC)

	references2 = np.einsum('ai,aj,j->ai',axes,axes,reference)
	references2 += np.einsum('a,ijk,ilm,al,m,ak->ai',np.cos(polar),LC,LC,axes,reference,axes)
	references2 += np.einsum('a,ijk,aj,k->ai',np.sin(polar),LC,axes,reference)

	#print(references-references2)


	#normals1 = np.einsum('aij,j->ai',rotationMatrices,np.array([0,0,1]))
	#normals2 = frame/np.array([Rs,Rs,Rs]).T
	#print(normals1-normals2)

	# with the bond orientations and the appriopriate reference vectors chosen,
	# the calculation of the 6-fold bond orientation order parameter is easy
	psi6 = np.zeros(N,dtype=complex)
	for k, bos in enumerate(bond_orientations):
		ref = references[k]
		v = vc[k]
		argument = np.array([np.arccos(np.dot(bo,ref)) for bo in bos])
		psi6[k] = np.sum(np.exp(6j*argument))/v

	return psi6, np.mean(np.abs(psi6))

def C6(frame):
	N = frame.shape[0]
	n, vc = findNeighbors(frame)
	
	psi6, psi6global = Psi6(frame)
	C6 = 0*vc

	for k, v in enumerate(vc):
		pi = psi6[k]
		pj = np.conjugate(psi6[n[k]==1])
		chi = np.abs(np.real(pi*pj))/np.abs(pi*pj)
		C6[k] = np.sum(chi>=0.32)

	return C6, np.mean(C6)/c6_hex(N), psi6, psi6global

if __name__=="__main__":

	nargs = len(sys.argv)
	if nargs <= 1:
		nfiles = len(glob.glob("output*.xyz"))
		index = np.arange(nfiles-1) 
		c = np.zeros(nfiles-1)
		p = np.zeros(nfiles-1)
		for i in index:
			frame = np.array(read_xyz_frame(f"output_{i+1}.xyz"))
			_, c[i], _, p[i] = C6(frame)

		fig, ax = plt.subplots()
		
		simArgument = json.load(open("simArgument.json",'r'))
		R = simArgument['radius']

		ax.set_title(f"Order Parameters for R = {R:.3f}")
		ax.set_ylim([0,1])
		ax.set_xlabel("sweeps")
		ax.scatter(index*simArgument['nsnap'],c, label = r"[$C_6$]")
		ax.scatter(index*simArgument['nsnap'],p, label = r"[$\psi_6$]")
		ax.legend()
		plt.savefig("Order Parameters",bbox_inches = 'tight')

	elif nargs == 2:
		framepath = sys.argv[1]
		frame = np.array(read_xyz_frame(framepath))
		n,vc = findNeighbors(frame)

		print(totalCharge(frame),averageCharge(frame))

		simArgument = json.load(open("simArgument.json",'r'))
		R = simArgument['radius']

		proj, _,_ = capPolarProjection(frame)
		fig,[ax1,ax2] = plt.subplots(1,2, figsize = (8,4))

		plt.suptitle(f"Projected snapshot R={R:.3f}")

		plt.tight_layout()#rect=[0, 0.03, 1, 1.2])
		ax1.set_aspect('equal', 'box')
		ax1.set_title("Particles Colored by Voronoi Charge")
		ax1.scatter(proj[:,0],proj[:,1],color=[getRGB(v) for v in Vc(frame,excludeborder=True)])


		localC, meanC, psi, globalPsi = C6(frame)
		print(meanC,globalPsi)

		ax2.set_aspect('equal', 'box')
		ax2.set_title(r"Particles Colored by $C_6$")
		ax2.scatter(proj[:,0],proj[:,1],c=localC,cmap='viridis')
		plt.savefig(f"{framepath}.png",bbox_inches = 'tight')

	elif nargs > 2:
		raise Exception("You entered too much stuff, fool")


# #for planar systems only
# def Ci6(coordinates, shellradius = 1.6):
# 	npart = coordinates.shape[0]
# 	i,j = np.mgrid[0:npart,0:npart]
# 	#dr = np.sqrt((coords[p,0]-coords[q,0])**2+(coords[p,1]-coords[q,1])**2)
# 	dr_vec = coordinates[i]-coordinates[j]
# 	dr_norm = np.linalg.norm(dr_vec,axis=-1)
# 	dr_norm[i==j] = 1e-3
	
# 	#to get the theta's we need an aribrtrary reference angle, we'll say (0,1) is our reference angle
# 	# we get the cosine with the dot product between dr and (0,1), and then divide it by the norm
# 	cosines = dr_vec[:,:,0]/dr_norm
# 	#fix nans:
# 	cosines[i==j]=0
	
# 	neighbors = dr_norm<shellradius
# 	neighbors[i==j]=False
# 	Nc = np.sum(neighbors,axis=-1)
# 	Nc[Nc==0]=1
	
# 	with np.errstate(divide='ignore'):

# 		argument = np.exp(6j * np.arccos(cosines))

# 		psi6 = np.array([(1/Nc[n])*np.sum(neighbors[n]*argument[n]) for n in range(npart)])

# 		psi6star = np.conjugate(psi6)

# 		chi6 = np.abs(np.real(psi6[i]*psi6star[j]))/np.abs(psi6[i]*psi6star[j])

# 		C6 = np.array([(1/6)*np.sum(neighbors[n]*(chi6[n]>0.32)) for n in range(npart)])

# 	return C6, Nc