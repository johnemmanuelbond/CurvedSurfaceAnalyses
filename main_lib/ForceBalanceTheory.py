# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 3, 2022
Compiled from several previous scripts/notebooks

This file contains the methods needed to predict
theoretical density curves using the osmotic
force balance continuum theory.

@author: Jack Bond
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from timeit import default_timer as timer


from .UnitConversions import field_k, get_a_eff

# Thermodynamic critical points for hard disc systems
eta_cp = 0.906  # close packed
eta_f = 0.69    # freezing point
eta_m = 0.71     # melting point

#equation of state
def Zf(eta):
	return (1 + (eta**2)/8) * (1 - eta)**-2

def dZf(eta):
	return (eta + 8)/(4 * (1 - eta)**3)

def Zs(eta):
	alpha = (eta_cp/eta) - 1
	return 2*alpha**-1 + 0.67*alpha + 1.9

def dZs(eta):
	alpha = (eta_cp/eta) - 1
	return (-2*alpha**-2+0.67) * (-eta_cp/eta**2)

# def Zc(eta):
# 	return Zf(eta_f)+(Zs(eta_m)-Zf(eta_f))/(eta_m-eta_f)*(eta-eta_f)

# def dZc(eta):
# 	return (Zs(eta_m)-Zf(eta_f))/(eta_m-eta_f)

"""
Via the osmotic pressure, energy is directly linked to density.
This funtion performs the integration to get an array of energies
from an array
"""
def energyFromForceBalance(eta_in):
	# array should be sorted in decreasing order
	# as we take our energy to be 0 at the highest density
	if np.all(np.diff(eta_in) <= 0):
		etas = np.copy(eta_in)
	else:
		etas = np.sort(eta_in)

	integrands = np.zeros(etas.size)
	dU = np.zeros(etas.size)

	coexist_mask = np.logical_and(etas>eta_f, etas<eta_m)
	fluid_mask = etas<=eta_f
	solid_mask = etas>=eta_m

	fluid = etas[fluid_mask]
	solid = etas[solid_mask]
	fluid_integrands = Zf(fluid)/fluid + dZf(fluid)
	solid_integrands = Zs(solid)/solid + dZs(solid)

	coexist = etas[coexist_mask]
	coexist_integrands = np.zeros(coexist_mask.sum())
	#coexist_integrands = Zc(coexist)/coexist + dZc(coexist)

	integrands[coexist_mask] = coexist_integrands;
	integrands[fluid_mask] = fluid_integrands;  # fluid region
	integrands[solid_mask] = solid_integrands;  # solid region

	widths = -np.diff(etas)
	heights = 0.5*(integrands[:-1] + integrands[1:])
	dU[1:] = heights * widths

	U = np.cumsum(dU)
	return U

"""
The energy can then be related to a distance via the functional form and strength
of the external field, in this case it is quadratic.
"""
def rFromUpf(u, params, extent = np.inf):
	k = field_k(params) #units of kT/(2a)^2
	assert k > 0, "field must be positive"
	rs = np.sqrt(u/k)

	#on spheres large arclength isn't allowed:
	rs[rs>extent] = extent
	return rs #units of 2a

"""
Count the number of particles on a surface given a density profile.
Assumes the particles are on a flat surface unless otherwise specified,
in which case rs become arclength--as opposed to radius-- and the bin 
areas need to change dramatically,
"""
def integrate_profile(rhos, rs, shellRadius=None):
	heights = 0.5*(rhos[1:]+rhos[:-1])
	if shellRadius == None:
		areas = np.pi*(rs[1:]**2-rs[:-1]**2)
	else:
		areas = (2*np.pi*shellRadius**2)*(np.cos(rs[:-1]/shellRadius)-np.cos(rs[1:]/shellRadius))
	counts = areas * heights
	return counts.sum()

"""
Given a density profile calculates the total number of particles in the field
Assumes the particles are on a flat surface unless otherwise specified, in which
case the bin areas need to change dramatically
"""
def integrate_histogram(rhos, rho_bin, shellRadius = None):
	if shellRadius == None:
		areas = np.pi*(rho_bin[1:]**2-rho_bin[:-1]**2)
	else:
		areas = (2*np.pi*shellRadius**2)*(np.cos(rho_bin[:-1]/shellRadius)-np.cos(rho_bin[1:]/shellRadius))
	counts = areas * rhos
	return counts.sum()

"""
By specifying a field strength and a target number of particles, we can
guess central densities until the resultant density distribution yields
a particle number consistent with the target.
"""
def profileFromFieldStrength(params, target_N, tol_e=-3, aeff=None, shellRadius=None):
	start = timer()
	
	a = params['particle_radius']
	if(aeff==None):
		aeff = get_a_eff(params)
	
#     stepsize = 1e-5
#     all_etas = np.linspace(eta_cp-stepsize,1e-3,num = int((eta_cp-stepsize-1e-3)/stepsize))
#     print(all_etas)
#     all_U = energyFromForceBalance(all_etas)
#     all_rs = rFromUpf(all_U,params)
	
	lo_eta = eta_f/3
	hi_eta = eta_cp
	
	guess_eta = (hi_eta-lo_eta)/2
	guess_N = 0
	
	k = 0

	while np.round(guess_N, decimals=-int(tol_e)) != target_N:
		eta_in = np.linspace(guess_eta,0.001,num=400)#all_etas[all_etas<guess_eta]
		U = energyFromForceBalance(eta_in)#all_U[all_etas<guess_eta]
		
		if(shellRadius==None):
			rs = rFromUpf(U,params)
		else:
			rs = rFromUpf(U,params, extent = np.pi*shellRadius)

		rhos = eta_in/((1/(2*a))**2 * np.pi*aeff**2) # #/(2a)^2
		
		guess_N = integrate_profile(rhos, rs, shellRadius=shellRadius)

		if guess_N > target_N: # too many particles, eta_0 is too high
			new_guess = guess_eta - (guess_eta - lo_eta)/2
			hi_eta = guess_eta
			guess_eta = new_guess
		else: # not enough particles, eta_0 is too low
			new_guess = guess_eta + (hi_eta - guess_eta)/2
			lo_eta = guess_eta
			guess_eta = new_guess
		if k > 1e4:
			print("Density Profile Timed Out")
			break
		k+=1

	end = timer()
	#print(rs)
	#print(f"bisection in {end-start}s")
		
	return eta_in, rs, guess_N

"""
By specifying a central density and a target number of particles, we can
guess field strengths until the resultant density distribution yields
a particle number consistent with the target.
"""
def profileFromCentralDensity(eta_c, params, target_N, tol_e=-3, aeff=None, shellRadius=None):
	start = timer()
	
	a = params['particle_radius']
	if(aeff==None):
		aeff = get_a_eff(params)

	eta_in = np.linspace(eta_c,0.001,num=400)
	U = energyFromForceBalance(eta_in)
	rhos = eta_in/((1/(2*a))**2 * np.pi*aeff**2) # #/(2a)^2

	pcopy = params.copy()

	lo_vpp = 1e-10
	hi_vpp = 30
	
	guess_vpp = pcopy['vpp']
	guess_N = 0
	
	while np.round(guess_N, decimals=-int(tol_e)) != target_N:
		
		if(shellRadius==None):
			rs = rFromUpf(U,pcopy)
		else:
			rs = rFromUpf(U,pcopy, extent = np.pi*shellRadius)
		
		guess_N = integrate_profile(rhos, rs, shellRadius=shellRadius)

		if guess_N < target_N: # not enough particles, profile needs to spread out, vpp is too high
			new_guess = guess_vpp - (guess_vpp - lo_vpp)/2
			hi_vpp = guess_vpp
			guess_vpp = new_guess
		else: # too many particles, profile needs to contact, vpp is too low
			new_guess = guess_vpp + (hi_vpp - guess_vpp)/2
			lo_vpp = guess_vpp
			guess_vpp = new_guess
		
		pcopy['vpp'] = guess_vpp
		
	end = timer()

	return eta_in, rs, guess_N, pcopy


"""calculates particle number density projected onto xy-plane
given a N x M x d array"""
def rho_hist(frames, furthest=None, bin_width=2):
	projected = frames[:,:,:2]
	fnum, pnum, _ = projected.shape #get number of frames and particles

	# converts projected coords into distances from center across sample
	dists = np.linalg.norm(projected, axis=-1).flatten()

	# if furthest is not defined, include 20% beyond farthest excursion
	if furthest is None:
		furthest = max(dists)*1.2

	hbin_edge = np.histogram_bin_edges(dists,
								   bins=int(furthest/bin_width),
								   range=(0, furthest))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2

	#annular area formula as defined below:
	#https://en.wikipedia.org/wiki/Annulus_(mathematics)
	area = np.pi*(hbin_edge[1:]**2 - hbin_edge[:-1]**2)
	#get count within each bin
	hval, hbin = np.histogram(dists, bins=hbin_edge)

	rho = hval / (fnum * area)
	return mids, rho, hbin

"""
calculates particle number density as a function of arclength on a sphere
given a N x M x d array and that sphere's radius--which is necessary for computing
bin areas.
"""
def spherical_rho_hist(frames, shellRadius, furthest=None, bin_width=2):
	fnum, pnum, _ = frames.shape #get number of frames and particles
	
	zcoords = frames[:,:,-1].flatten()
	zcoords[zcoords>shellRadius]=shellRadius
	zcoords[zcoords<-1*shellRadius] = -1*shellRadius
	
	arc_from_top = shellRadius*np.arccos(zcoords/shellRadius)
	#arc_from_bot = shellRadius*np.arccos(-1*zcoords/shellRadius)
	arclengths = arc_from_top#np.minimum(arc_from_top,arc_from_bot)
	
	# if furthest is not defined, include 20% beyond farthest arclength
	if furthest is None:
		furthest = max(arclengths)*1.2
	
	hbin_edge = np.histogram_bin_edges(arclengths,
								   bins=int(furthest/bin_width),
								   range=(0, furthest))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2
	
	#so I did this calculus on paper, it would be helpful to find a reference for it.
	area = (2*np.pi*shellRadius**2)*(np.cos(hbin_edge[:-1]/shellRadius)-np.cos(hbin_edge[1:]/shellRadius))
	
	hval, hbin = np.histogram(arclengths, bins=hbin_edge)
	
	rho = hval / (fnum * area)
	
	return mids, rho, hbin

"""
Main method runs a few theortical calculations
"""
if __name__=="__main__":
	#experimental parameters in SI units
	params = {#solution characteristics
			'temperature': 298,     # [K]
			'rel_permittivity': 78,     # unitless
			'ion_multiplicity': 1,      # unitless
			'debye_length': 10e-9,      # [m]
			#particle characteristics
			'particle_radius': 1.4e-6,  # [m]
			'surface_potential': -75e-3,    # [V]
			'fcm':  -0.2287,        # unitless
			#field characteristics
			'vpp': 14,          # [V]
			'dg': 100e-6,           # [m]
			}

	print(get_a_eff(params))

	testN = 300

	vpps = np.array([1.5, 2.3, 3, 10])
	Rs = np.array([7.471, 11.201, 23.330, 32.661, 93.313])

	etas = np.linspace(eta_cp-0.001, 0.001, 40000)

	def Z(eta):
		if (eta <= eta_f):
			return Zf(eta)
		elif (eta >= eta_m):
			return Zs(eta)
		else:
			return 0#Zc(eta)

	fig, ax = plt.subplots()
	ax.set_xlim([0,1])
	ax.set_ylim([0,30])
	ax.set_ylabel("Z")
	ax.set_xlabel("$\eta$")
	ax.set_title("Equation of State")
	ax.plot(etas, [Z(eta) for eta in etas])
	plt.show()

	for v in vpps:
		params['vpp']=v
		a = params['particle_radius']*1e6
		
		fig, ax = plt.subplots()
		ax.set_ylim([0,1])
		ax.set_xlabel("Arclength [$\mu$m]")
		ax.set_ylabel("$\eta_{\text{eff}}$")
		ax.set_title(f"Theoretical Density Profiles for $V_{{pp}}$ = {v}V")
		
		eta_th, rs, guess_N = profileFromFieldStrength(params,testN, tol_e=-2)
		ax.plot(rs*2*a, eta_th,ls='--', label=f'Flat Case')
		
		for R in Rs:
			#eta_th, rs, guess_N = sphericalBisectionProfile(params,100, R, tol_e=-3)
			eta_th, rs, guess_N = profileFromFieldStrength(params,testN, shellRadius = R, tol_e=-2)
			ax.plot(rs*2*a, eta_th,ls='--', label=f'Radius: {R*2*a:.2f} [$\mu m$]')
		
		#ax.legend(bbox_to_anchor=(1.5,1))
		ax.legend()
		plt.show()


	
	eta_0 = 0.85
	print(f"eta = {eta_0} Voltages:")

	a = params['particle_radius']*1e6

	fig, ax = plt.subplots()
	ax.set_ylim([0,1])
	ax.set_xlabel("Arclength [$\mu$m]")
	ax.set_ylabel("$\eta_{\text{eff}}$")
	ax.set_title(f"Theoretical Density Profiles for $\eta_0$ = {eta_0}")

	eta_th, rs, guess_N, pcopy = profileFromCentralDensity(eta_0, params,testN, tol_e=-2)
	ax.plot(rs*2*a, eta_th,ls='--', label=f'Flat Case')
	print(f"Flat Case: {pcopy['vpp']} V")

	for R in Rs:
		#eta_th, rs, guess_N = sphericalBisectionProfile(params,100, R, tol_e=-3)
		eta_th, rs, guess_N, pcopy = profileFromCentralDensity(eta_0, params,testN, shellRadius = R, tol_e=-2)
		ax.plot(rs*2*a, eta_th,ls='--', label=f'Radius: {R*2*a:.2f} [$\mu m$]')
		print(f"{R} [2a]: {pcopy['vpp']} V")

	#ax.legend(bbox_to_anchor=(1.5,1))
	ax.legend()
	plt.show()