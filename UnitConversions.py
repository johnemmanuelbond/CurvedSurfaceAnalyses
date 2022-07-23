# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 3, 2022
Compiled from several previous scripts/notebooks

This file contains the methods needed to smoothly convert between SI
units and the natural units of the simulation: 2a, kT

@author: Jack Bond
"""

import numpy as np
import scipy as sp
from scipy import integrate
from timeit import default_timer as timer

# General physical constants
kb = 1.380e-23 # [J/K] Boltzmann constant
e = 1.602177e-19 #[C] elementary charge
eps = 8.854e-12 #[F/m] or [J/(m*V**2)], permittivity of free space

"""
These methods take in a dictionaty of experimental parameters in SI
and returns the three quantities needed to specify an MC simulation
in their natural units. Also, since the vpp can change within a batch
of data, there is a method to get the vpp from a known field k (in natural units).
"""
def yukawa_prefactor(params):
	a = params['particle_radius']            # [m]
	psi = params['surface_potential']        # [V]
	kT = kb*params['temperature']            # [J]
	ze = params['ion_multiplicity']*e        # [C]
	rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n

	#below gives bpp in joules [J]
	bpp = 32 * np.pi * rel_eps * a * ((kT/ze)**2) * np.tanh((ze*psi) / (4*kT))**2
	return bpp/kT  # converts from [J] to units of [kT]

def field_k(params):
	a = params['particle_radius']            # [m]
	kT = kb*params['temperature']            # [J]
	rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n

	#calculating field spring constant
	field_strength = (params['vpp']/(params['dg']**2))**2
	#below gives field spring constant with units of [J*m**-2]
	field_prefactor = -4 * np.pi * rel_eps * (a**3) * params['fcm'] * field_strength

	return field_prefactor/(kT * (2*a)**-2)  # converts [J*m**-2] to units of [kT*(2a)**-2]

def length_scale(params):
	a = params['particle_radius']            # [m]
	return params['debye_length']/(2*a) # converts [m] to [2a]

def getSimInputTerms(params):
	return yukawa_prefactor(params), field_k(params), length_scale(params)

def vpp(field_k, params):
	a = params['particle_radius']            # [m]
	kT = kb*params['temperature']            # [J]
	rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n

	field_prefactor = field_k*(kT * (2*a)**-2) # converts [kT*(2a)**-2] to units of [J*m**-2]

	#below gives field strength from a spring constant with units of [J*m**-2]
	field_strength = field_prefactor/(-4 * np.pi * rel_eps * (a**3) * params['fcm'])

	return np.sqrt(field_strength)*(params['dg']**2)


"""
Methods to convert between density and effective area fraction,
which hold the same information anyway
"""
def rho(eta, a_eff):
	return eta/(np.pi * a_eff**2)  #in units of 1/[a_eff**2]

def eta(rho, a_eff):
	return rho*(np.pi*a_eff**2) #unitless



#based off literature perturbation theory:
def getAEff(params):
	start = timer()
	a = params['particle_radius']
	kappa = 1/params['debye_length']
	Y_pf = yukawa_prefactor(params)
	
	integrand = lambda r: 1-np.exp(-1*Y_pf*np.exp(-1*kappa*r))
	
	debye_points = np.arange(5)/(kappa)
	
	first, fErr = sp.integrate.quad(integrand, 0, 1000/kappa, points=debye_points)
	second, sErr = sp.integrate.quad(integrand, 1000/kappa, np.inf)
	
	aeff = (a + 1/2*(first+second))
	
	end = timer()
	#print(end-start)
	return aeff

def R_for_etaeff(pnum, eta_eff, aeff_a_ratio):
    return (1/4)*np.sqrt(pnum/eta_eff)*aeff_a_ratio