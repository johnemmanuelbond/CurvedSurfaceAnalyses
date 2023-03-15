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
e = 1.602177e-19 # [C] elementary charge
eps = 8.854e-12 # [F/m] or [J/(m*V**2)], permittivity of free space
Na = 6.0221408e23 # [1/mol] Avogadro's number

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

def kappa(params):
	if 'debye_length' in params:
		return 1/length_scale(params)

	ze = params['ion_multiplicity']*e        # [C] electrolyte charge
	kT = kb*params['temperature']            # [J]
	rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n
	C = params['electrolyte_concentration']  # [mol/m^3]

	#assuming a symmetric electrolyte with
	return np.sqrt(2*(ze**2)*(C*1000)*Na/(rel_eps*kT)) #[1/m]

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

if __name__=="__main__":

        params = {#solution characteristics
                'temperature': 298,             # [K]
                'rel_permittivity': 78,         # unitless
                'ion_multiplicity': 1,       # unitless
                'debye_length': 10e-9,          # [m]
                'viscosity': 0.8931e-3,             # [Pa*s]
                #particle characteristics
                'particle_radius': 1.4e-6,      # [m]
                'particle_density': 1980,        # [kg/m^3]
                'surface_potential': -75e-3,    # [V]
                'fcm':  -0.2287,                # unitless
                #field characteristics
                'vpp': 0.0,#1.0                 # [V]
                'dg': 100e-6,                   # [m]
        }

        Ypf, _, debye = getSimInputTerms(params)
        kappa = 1/debye
        a_hc, a_eff_app = params['particle_radius'], getAEff(params)
        
        print(f"Energy Prefactor: {Ypf:.2f}[kT]")
        print(f"aeff: {a_eff_app:.5e}[um]\n2aeff: {a_eff_app/a_hc:.5f}[2a]")

        #what happens if we assume Ypf is divided by kappa in lammps to get the energy:
        params_adj = params
        params_adj['rel_permittivity']*=1/kappa #Ypf ~ rel_perm without affecting other parts of the calculation
        Ypf, _, _ = getSimInputTerms(params_adj)
        a_hc, a_eff_corr = params['particle_radius'], getAEff(params)
        
        print(f"Energy Prefactor: {Ypf:.2f}[kT]")
        print(f"aeff: {a_eff_corr:.5e}[um]\n2aeff: {a_eff_corr/a_hc:.5f}[2a]")

        print(f"area fraction correction factor: {(a_eff_corr/a_eff_app)**2:.3f}")

        etas_app = np.array([0.1,0.3,0.5,0.6,0.65,0.69,0.73,0.75])
        etas_corr = (a_eff_corr/a_eff_app)**2*etas_app

        print(etas_app)
        print(etas_corr)
