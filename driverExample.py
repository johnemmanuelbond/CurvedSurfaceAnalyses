import sys, os

Jack_Analysis_dir = "C:/Users/johne/MolecularMonteCarlo/AnalysisEngine"
Marcc_Analysis_dir = "~/bin/CurvedSurfaceAnalyses"

sys.path.insert(1,Jack_Analysis_dir)

import UnitConversions as units
import DriverUtilities as util

import numpy as np

from datetime import datetime

params = {#solution characteristics
		'temperature': 298,             # [K]
		'rel_permittivity': 78,         # unitless
		'ion_multiplicity': 1,          # unitless
		'debye_length': 10e-9,          # [m]
		#particle characteristics
		'particle_radius': 1.4e-6,      # [m]
		'surface_potential': -75e-3,    # [V]
		'fcm':  -0.2287,                # unitless
		#field characteristics
		'vpp': 0.4,                     # [V]
		'dg': 100e-6,                   # [m]
		}

simArgument = {#arguments for a simulation
		'nsweeps': 1000,
		'nsnap': 10,
		'npart': 100,
		'radius': 100,            #[2a]
		'temp': 1,               #[kT]
		'rc': 5,        #[2a]
		'length_scale': 1e-2,           #[2a]
		'fieldk': 1,                   #[kT/(2a)^2]
		'a': 6000,                      #[kT]
		'start_from_config': False,
		}

#define test cases for high-curvature
Rs = np.array([4, 5, 6, 100])
vs = np.array([2.864980697703186,3.0020332337125577,3.0670166016318303,3.2001399994576474])

# we like to run a bunch of random copies of the simulation for sampling purposes
nseeds = 3

#starting the log
now = datetime.now()
dt = now.strftime("%d/%m/%Y %H:%M:%S")
log = open("log.txt", "w")
log.write(f"{dt}:: Simulations Initiated\n")

#using the independent sample time to fix the nsnap frequency
simArgument = util.fixFrequency(simArgument,util.Liquid_Ind_Sample_Time)

for j, R in enumerate(Rs):
	#setting up the corrected dictionaries
	params['vpp'] = vs[j]

	yukawa_prefactor, field_k, length_scale = units.getSimInputTerms(params)

	simArgument['radius'] = float(R)
	simArgument['a'] = yukawa_prefactor
	simArgument['fieldk'] = field_k
	simArgument['length_scale'] = length_scale

	#logging the simulations
	log.write(f"\t{nseeds} copies of\n")
	for arg in simArgument:
		log.write(f"\t\t {arg} {simArgument[arg]}\n")

	#running the sims
	for i in np.arange(nseeds):
			fldr = f"R_{R:.3f}_N_{simArgument['npart']}_snapshots_{i}"
			util.runPCSim(simArgument,params,fldr)
			#util.runMarccSim(simArgument,fldr)
			#util.runPCSim(simArgument,fldr,exe=utils.exe_Jack_Flat)

log.write("\n")
log.close()

#performing the analysis
anapath = Jack_Analysis_dir+"/DensityPlotter.py"
os.system(f"python {anapath}")
