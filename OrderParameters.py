import UnitConversions as units
#from visualize import getRGB
from FileHandling import read_xyz_frame

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi#, geometric_slerp
from scipy.signal import find_peaks

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

def radialDistributionFunction(frame, info=None):
	npart = frame.shape[0]
	

	if info == None:
		dr_norm = sp.spatial.distance.squareform(pdist(frame))
		i,j = np.mgrid[0:npart,0:npart]
		dists = dr_norm[i<j].flatten()
		info = {
			"distance_matrix": dr_norm,
			"distance_list": dists
			}
	else:
		dists = info["distance_list"]

	hbin_edge = np.histogram_bin_edges(dists,
								   bins=int(npart),
								   range=(0, 3*min(dists)))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2

	#print(min(info['distance_list']))

	hval, hbin = np.histogram(dists, bins=hbin_edge)

	hist = np.array([mids,hval])

	#minimumDistance = mids[hval!=0][0]

	#relevant = mids<4*minimumDistance

	#mids = mids[relevant]
	#hval = hval[relevant]

	peaks, _ = find_peaks(hval,prominence=10)

	# shellRadius = (mids[peaks[1]]+mids[peaks[0]])/2
	spacing = mids[peaks[0]]
	info["particle_spacing"] = spacing

	relevant = (mids>spacing)*(mids<2*spacing)
	relevantMids = mids[relevant]
	relevantHval = hval[relevant]
	shellRadius = relevantMids[np.argmin(relevantHval)]

	info["first_coordination_shell"] = shellRadius

	return hist, info

"""
Given a frame or set of frames, computed the average radial distribution function
source: general_analysis, 7/23/22
author: Alex yeh
"""
def g_r(coords, shell_radius=None, bin_width=0.1):
    """calculates the pair distribution function from the given coordinates"""
    if shell_radius is None:
        # get mean radius over run
        shell_radius = np.linalg.norm(coords, axis=-1).mean()
        
    fnum, pnum, _ = coords.shape
    
    allrs = np.zeros((fnum, (pnum*(pnum-1)//2)))
    for t, frame in enumerate(coords):
        for i, p1 in enumerate(frame):
            for j in range(i+1, pnum):
                flat_idx = pnum*i - i*(i+1)//2 + j - i - 1
                cos_dist = np.dot(frame[i], frame[j])/(shell_radius**2)
                if cos_dist>1: cos_dist=1
                if cos_dist<-1: cos_dist=-1
                allrs[t, flat_idx] = shell_radius*np.arccos(cos_dist)
    bins = np.histogram_bin_edges(allrs[0],
                                  bins = int(np.pi*shell_radius/bin_width),
                                  range = (0, np.pi*shell_radius))
    angle_bins = bins/shell_radius
    width = bins[1] - bins[0]
    mids = bins[:-1] + width/2
    hval = np.zeros_like(mids)
    
    counts, _ = np.histogram(allrs, bins=bins)
    vals = counts/(fnum*allrs.shape[1]) * 2/(np.cos(angle_bins[:-1]) - np.cos(angle_bins[1:]))
    return vals, mids, bins

def firstCoordinationShell(frame, info=None):
	if(info == None):
		_, info = radialDistributionFunction(frame, info)
	return info["first_coordination_shell"]

#a simple coordination number 
def Nc(frame, shellradius = (1.44635/1.4)*0.5*(1+np.sqrt(3))):
	npart = frame.shape[0]
	i,j = np.mgrid[0:npart,0:npart]
	dr_norm = sp.spatial.distance.squareform(pdist(frame))
	
	neighbors = dr_norm<shellradius
	neighbors[i==j]=False
	Nc = np.sum(neighbors,axis=-1)

	return Nc

#coordination number based of voronoi triangulation
def Vc(frame,excludeborder=False,R=None,tol=1e-6):
	minZ = min(frame[:,2])
	
	if R == None:
		radius = np.mean(np.linalg.norm(frame,axis=1))
	else:
		radius = R
	sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
	Vc = np.zeros(frame.shape[0])
	for i, region in enumerate(sv.regions):
		Vc[i] = len(region)
		if(excludeborder):
			for v in sv.vertices[region]:
				if(v[2]<minZ):
					Vc[i]+=-1
					#Vc[i]=-1
	return Vc

#returns an Nx3 array of rgb values based on the voronoi tesselation of a frame
def voronoi_colors(frame,v=None,tol=1e-6):
    if type(v)==type(None):
    	v = Vc(frame, excludeborder=False,tol=tol)
    #print(np.sum(6-v))
    #print(np.sum(np.abs(6-v)))
    colors = np.array([[0.6,0.6,0.6] for _ in v])
    greens = np.array([[0,0.5*vi/6,0.2] for vi in v])
    reds = np.array([[1-0.5*vi/6,0,0.2+0] for vi in v])
    colors[v>6] = greens[v>6]
    colors[v<6] = reds[v<6]
    return colors

#point-density based on the area of voronoi polygons on a frame
def rho_voronoi(frame,excludeborder=False,R=None,tol=1e-6):
	minZ = min(frame[:,2])
	
	if R == None:
		radius = np.mean(np.linalg.norm(frame,axis=1))
	else:
		radius = R
	sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
	V_rho = np.zeros(frame.shape[0])
	for i, area in enumerate(sv.calculate_areas()):
		V_rho[i] = 1/area
	return V_rho

#point-density based on the area of voronoi polygons INCLUDING NEAREST NEIGHBORS on a frame
def rho_voronoi_shell(frame,excludeborder=False,R=None,tol=1e-6):
	minZ = min(frame[:,2])
	
	if R == None:
		radius = np.mean(np.linalg.norm(frame,axis=1))
	else:
		radius = R

	sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
	_,info = radialDistributionFunction(frame)
	neighbors = np.array(1*(info["distance_matrix"]<=info["first_coordination_shell"]))

	V_rho = np.zeros(frame.shape[0])
	areas = sv.calculate_areas()
	for i, nei in enumerate(neighbors):
		As = areas[np.where(nei!=0)[0]]
		area = As.sum()
		nshell = As.size
		V_rho[i] = nshell/area
	return V_rho

#returns an Nx3 array of rgb values based on the voronoi tesselation of a frame
def density_colors(frame,rhos=None,aeff = 0.5,tol=1e-6):
    if type(rhos) == type(None):
    	rhos = rho_voronoi(frame, excludeborder=False,tol=tol)
    #print(np.sum(6-v))
    #print(np.sum(np.abs(6-v)))
    rho_cp = 0.9067/(np.pi*aeff**2)
    rho_fl = 0.69/(np.pi*aeff**2)
    rho_mean = rhos.mean()
    scale = (rhos-rho_mean)/(rho_cp-rho_fl)
    colors = np.array([[0.5-s,0.5+s,0.5] for s in scale])
    return colors


def shareVoronoiVertex(sv, i, j):
	vertices_i = sv.regions[i]
	vertices_j = sv.regions[j]

def findScarsCarefully(frame,tol=1e-6):
	N = frame.shape[0]
	radius = np.mean(np.linalg.norm(frame,axis=1))
	sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
	qs = np.array([6-len(region) for region in sv.regions])

	shared_vertices = np.array([[np.sum(np.isin(r1,r2)) for r1 in sv.regions] for r2 in sv.regions])
	charged_pairs = np.abs(np.array([qs for _ in qs])*np.array([qs for _ in qs]).T)

	charged_neighbors = np.array((shared_vertices>=1)*(charged_pairs>0))

	scars = []

	for i, ptcl in enumerate(charged_neighbors):
		links = np.sort(np.where(ptcl!=0)[0])
		if links.size !=0:
			uniquePtcls = True
			for j, scar in enumerate(scars):
				if(np.any(np.isin(scar, links))):
					newscar = np.unique(np.append(scar,links))
					scars.pop(j)
					scars.append(newscar)
					links = newscar
					uniquePtcls = False
			if(uniquePtcls):
				scars.append(links)

	scarCharges = np.array([np.sum(qs[scar]) for scar in scars])

	return scars, scarCharges


def findScars(frame,tol=1e-6,coordinationShells=1.1):

	charge = 6-Vc(frame,tol=tol)
	_,info = radialDistributionFunction(frame)

	neighbors = np.array(1*(info["distance_matrix"]<=coordinationShells*info["first_coordination_shell"]))
	#neighbors = neighbors-np.eye(len(charge))
	charged_pairs = np.abs(np.array([charge for _ in charge])*np.array([charge for _ in charge]).T)

	charged_neighbors = np.array(neighbors*charged_pairs)

	scars = []
	
	# pairs = np.unique(np.sort(np.array(np.where(charged_neighbors)).T,axis=-1),axis=0)
	# for pair in pairs:
	# 	uniquePtcls = True
	# 	for i, scar in enumerate(scars):
	# 		if(np.any(np.isin(scar, pair))):
	# 			newscar = np.unique(np.append(scar,pair))
	# 			scars.pop(i)
	# 			scars.append(newscar)
	# 			uniquePtcls = False
	# 	if(uniquePtcls):
	# 		scars.append(pair)
	
	for i, ptcl in enumerate(charged_neighbors):
		links = np.sort(np.where(ptcl!=0)[0])
		if links.size !=0:
			uniquePtcls = True
			for j, scar in enumerate(scars):
				if(np.any(np.isin(scar, links))):
					newscar = np.unique(np.append(scar,links))
					scars.pop(j)
					scars.append(newscar)
					links = newscar
					uniquePtcls = False
			if(uniquePtcls):
				scars.append(links)

	scarCharges = np.array([np.sum(charge[scar]) for scar in scars])

	return scars, scarCharges

def ScarNumber(frame,tol=1e-6):
	Sc = np.array([None for _ in range(frame.shape[0])])

	scars, scarCharges = findScars(frame,tol=tol)

	for i, scar in enumerate(scars):
		Sc[scar] = scarCharges[i]

	return Sc

#returns an Nx3 array of rgb values based on the net voronoi charge of a frame
def scar_colors(frame,s=None,tol=1e-6):
    if type(s) == type(None):
    	s = ScarNumber(frame,tol=tol)
    colors = np.zeros((s.size,3))
    for i,si in enumerate(s):
    	if si == None:
    		colors[i] = np.array([0.6,0.6,0.6])
    	elif si == 0:
    		colors[i] = np.array([0.2,0.2,0.6])
    	elif si > 0:
    		colors[i] = np.array([0.5+(si-1)/2,0.2,0.2])
    	elif si < 0:
    		colors[i] = np.array([0.2,0.5+(1-si)/2,0.2])

    return colors


def shells(pnum):
    """from particle number, calculate number of shells assuming hexagonal crystal"""
    # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
    return -1/2 + np.sqrt((pnum-1)/3 + 1/4)

def c6_hex(pnum):
    """returns C6 for a hexagonal cluster of the same size"""
    # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
    s = shells(pnum)
    return 6*(3*s**2 + s)/pnum

# a little depricated atm
def findNeighbors(frame, info = None,tol=1e-6):
	N = frame.shape[0]
	i,j = np.mgrid[0:N,0:N]

	vc = Vc(frame,tol=tol)
	
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

		rad_dist, info = radialDistributionFunction(frame)

		coord_shell = info["first_coordination_shell"]
		a = info["particle_spacing"]


		nc = Nc(frame, coord_shell)
		#print(nc[np.where(vc!=nc)],vc[np.where(nc!=vc)], (6-nc[np.where(vc!=nc)])-(6-vc[np.where(nc!=vc)]))

		print(f"Voronoi Tesselation: total charge: {np.sum(6-vc)} excess charge: {0.5*(np.sum(np.abs(6-vc))/12-1)}")
		print(f"Coordination Number: total charge: {np.sum(6-nc)} excess charge: {0.5*(np.sum(np.abs(6-nc))/12-1)}")

		start = timer()
		qs = 6-Nc(frame,shellradius=coord_shell)
		end = timer()
		print(f"charges computed in {end-start}s")


		[rs, hval] = rad_dist
		print(f"First Coordination Shell: {coord_shell}")
		print(f"R/a: {np.mean(np.linalg.norm(frame,axis=-1))/a}")
		fig, ax = plt.subplots()
		ax.plot(rs,hval,lw=0.6,c="black")
		plt.show()

		findScars(frame)

		# simArgument = json.load(open("simArgument.json",'r'))
		# R = simArgument['radius']

		# proj, _,_ = capPolarProjection(frame)
		# fig,[ax1,ax2] = plt.subplots(1,2, figsize = (8,4))

		# plt.suptitle(f"Projected snapshot R={R:.3f}")

		# plt.tight_layout()#rect=[0, 0.03, 1, 1.2])
		# ax1.set_aspect('equal', 'box')
		# ax1.set_title("Particles Colored by Voronoi Charge")
		# ax1.scatter(proj[:,0],proj[:,1],c=nc)


		# localC, meanC, psi, globalPsi = C6(frame)
		# print(meanC,globalPsi)

		# ax2.set_aspect('equal', 'box')
		# ax2.set_title(r"Particles Colored by $C_6$")
		# ax2.scatter(proj[:,0],proj[:,1],c=localC,cmap='viridis')
		# plt.savefig(f"{framepath}.png",bbox_inches = 'tight')

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
