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


def yukawa_prefactor(params):
    """
    Takes in a dictionary of experimental quantities in  SI units.
    Returns the prefactor on a screned electrostatic repulsion in kT units
    Depends on the particle size and surface potential, the temperature,
    and the permittivity and ion multiplicity of the screening solution.
    """

    a = params['particle_radius']            # [m]
    psi = params['surface_potential']        # [V]
    kT = kb*params['temperature']            # [J]
    ze = params['ion_multiplicity']*e        # [C]
    rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n

    #below gives bpp in joules [J]
    bpp = 32 * np.pi * rel_eps * a * ((kT/ze)**2) * np.tanh((ze*psi) / (4*kT))**2

    return bpp/kT  # converts from [J] to units of [kT]


def kappa(params):
    """
    Takes in a dictionary of experimental quantities in  SI units.
    Returns the (unitless) decay constant for screened electrostatic repulsion 
    If params includes a debye length then it only needs that
    Otherwise, depends on the temperature, and the permittivity, ion
    multiplicity, and electrolyte concentration of the solution.
    """

    if 'debye_length' in params:
        return 2*params['particle_radius']/params['debye_length']

    kT = kb*params['temperature']            # [J]
    ze = params['ion_multiplicity']*e        # [C] electrolyte charge
    rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n
    C = params['electrolyte_concentration']  # [mol/L]

    #assuming a symmetric electrolyte
    return np.sqrt(2*(ze**2)*(C*1000)*Na/(rel_eps*kT)) #[1/m]


def ion_conc(params):
    """
    Takes in a dictionary of experimental quantities in  SI units.
    If unknown, returns the concentration of ions in solution.
    Depends on the temperature, and the permittivity, ion
    multiplicity, and debye length of the solution.
    """
    if 'electrolyte_concentration' in params:
        return params['electrolyte_concentration']

    kT = kb*params['temperature']            # [J]
    ze = params['ion_multiplicity']*e        # [C] electrolyte charge
    rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n
    debye_length = params['debye_length']    # [m]

    #assuming a symmetric electrolye
    return 1/1000 * rel_eps*kT*(1/debye_length**2)/(2*(ze**2)*Na)


def field_k(params):
    """
    Takes in a dictionary of experimental quantities in  SI units.
    Returns the prefactor on a quadratic external field of form k*r^2
    confining a particle to 0,0 in units of kT/(2a)^2
    Depends on the particle size, the temperature, solution permittivity
    and the voltage drop and gap distance of the electrode.
    """

    a = params['particle_radius']            # [m]
    kT = kb*params['temperature']            # [J]
    rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n

    #calculating field spring constant
    field_strength = (params['vpp']/(params['dg']**2))**2
    #below gives field spring constant with units of [J*m**-2]
    field_prefactor = -4 * np.pi * rel_eps * (a**3) * params['fcm'] * field_strength

    return field_prefactor/(kT * (2*a)**-2)  # converts [J*m**-2] to units of [kT*(2a)**-2]


def vpp(field_k, params):
    """
    Takes in a dictionary of experimental quantities in  SI units. As well as a
    given prefactor on a quadratic field
    Returns the voltage drop for said quadratic field in SI units
    Depends on the particle size, the temperature, solution permittivity
    and gap distance of the electrode.
    """
    
    a = params['particle_radius']            # [m]
    kT = kb*params['temperature']            # [J]
    rel_eps = params['rel_permittivity']*eps # [F/m] permittivity of sol'n

    field_prefactor = field_k*(kT * (2*a)**-2) # converts [kT*(2a)**-2] to units of [J*m**-2]

    #below gives field strength from a spring constant with units of [J*m**-2]
    field_strength = field_prefactor/(-4 * np.pi * rel_eps * (a**3) * params['fcm'])

    return np.sqrt(field_strength)*(params['dg']**2)


def get_a_eff(params):
    """
    Takes in a dictionary of experimental quantities in  SI units.
    Returns the effective radius for a hard disc interaction with the same
    second virial coeffecient as a screened electrostatic interaction.
    Depends on the particle size and surface potential, the temperature,
    and the permittivity, ion multiplicity, and ion concentration/debye length
    of the screening solution.
    """
    start = timer()
    a = params['particle_radius']
    kap = kappa(params['debye_length'])
    Y_pf = yukawa_prefactor(params)
    
    integrand = lambda r: 1-np.exp(-1*Y_pf*np.exp(-1*kap*r))
    
    debye_points = np.arange(5)/(kap)
    
    first, fErr = sp.integrate.quad(integrand, 0, 1000/kap, points=debye_points)
    second, sErr = sp.integrate.quad(integrand, 1000/kap, np.inf)
    
    aeff = (a + 1/2*(first+second))
        
    end = timer()
    #print(end-start)
    return aeff


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

    Ypf = yukawa_prefactor(params)
    a_hc, a_eff_app = params['particle_radius'], get_a_eff(params)
    
    print(f"Energy Prefactor: {Ypf:.2f}[kT]")
    print(f"aeff: {a_eff_app:.5e}[um]\n2aeff: {a_eff_app/a_hc:.5f}[2a]")

    #what happens if we assume Ypf is divided by kappa in lammps to get the energy:
    params_adj = params
    params_adj['rel_permittivity']*=1/kappa #Ypf ~ rel_perm without affecting other parts of the calculation
    Ypf = yukawa_prefactor(params_adj)
    a_hc, a_eff_corr = params['particle_radius'], get_a_eff(params_adj)
    
    print(f"Energy Prefactor: {Ypf:.2f}[kT]")
    print(f"aeff: {a_eff_corr:.5e}[um]\n2aeff: {a_eff_corr/a_hc:.5f}[2a]")

    print(f"area fraction correction factor: {(a_eff_corr/a_eff_app)**2:.3f}")

    etas_app = np.array([0.1,0.3,0.5,0.6,0.65,0.69,0.73,0.75])
    etas_corr = (a_eff_corr/a_eff_app)**2*etas_app

    print(etas_app)
    print(etas_corr)
