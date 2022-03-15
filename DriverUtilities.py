import numpy as np
import random

import os, glob
import json
from timeit import default_timer as timer

from datetime import datetime

#Flat-Case executables
exe_Marcc_Flat = "~/bin/plane_yukawa_shell"
exe_Jack_Flat = "C:/Users/johne/MolecularMonteCarlo/FlatSourceCode/plane_yukawa_field/build/bin/plane_yukawa_shell.exe"

#Spherical Shell executables
exe_Marcc_Curved = "~/bin/yukawa_shell"
exe_Jack_Curved =  "C:/Users/johne/MolecularMonteCarlo/SphericalCapSourceCode/shell_yukawa_field/build/bin/yukawa_shell.exe "

#the autocorrelation time tells you how far apart to take your
#snapshots such that they are statistically independent
#the values below are guesses based on fitting the autocorrelation
#to an exponential decay post-simulation
Solid_Ind_Sample_Time = 400
Liquid_Ind_Sample_Time = 50
#we want a nice integer number of samples, airing on the side of too few,
#then we get a snapshot frequency that's safe if the the number of sweeps
#isn't divisible by the independent sampling time.
def fixFrequency(simArgument, indSampleTime):
	nsamples = np.floor(simArgument['nsweeps']/indSampleTime)
	nsnapfreq = int(simArgument['nsweeps']//nsamples)
	simArgument['nsnap'] = nsnapfreq
	return simArgument

#code for dumping the relevant dictionaries into files
def dumpDictionaryJSON(dic, name):
	file = open(name+".json","w")
	file.write(json.dumps(dic))
	file.close()

def dumpDictionaryYAML(dict, name):
	return None

#kinda brute-forced atm
def getRandomInputFile(shellRadius):
	files = glob.glob(f"*R_{shellRadius:.3f}_Solids*/*")
	return random.choice(files)

#turns a dictionary of the MC code's lyra keys into a string
#which the command line can run
def getSimcmd(simArgument, exepath, isPC = False):
	cmd = "time " + exepath
	if isPC:
		cmd = exepath

	for arg in simArgument:
		if arg == "a" or arg == "i":
			cmd += f" -{arg} {simArgument[arg]}"
		elif arg == "start_from_config":
			if simArgument[arg]:
				cmd+= f" --datapath {getRandomInputFile(simArgument['radius'])}"
		else:
			cmd += f" --{arg} {simArgument[arg]}"

	return cmd

#runs a simulation using Marcc's parallelism
def runMarccSim(simArgument, params, fldr, exe = exe_Marcc_Curved):
	pwd = os.getcwd()
	cmd = getSimcmd(simArgument,exe)
	print("Foldername :", fldr)
	# Commands to create folder and run jobs

	os.system("mkdir "+fldr)

	os.system("cp sub.sh sub_edit.sh")

	fin = open("sub_edit.sh", "rt")
	data = fin.read()
	data = data.replace('xxx',f"{pwd}/{fldr}")
	data = data.replace('yyy',cmd)
	fin.close()

	fin = open("sub_edit.sh", "wt")
	fin.write(data)
	fin.close()

	#os.system('echo "'+cmd+'" >> sub_edit.sh')
	os.system("mv sub_edit.sh " + fldr + "/sub.sh")

	os.chdir(fldr)
	os.system("sbatch sub.sh;")

	dumpDictionaryJSON(simArgument,"simArgument")
	dumpDictionaryJSON(params,"params")

	os.chdir("..")

	return

#runs a simulation using a version of the code compiled on a windows machine
def runPCSim(simArgument, params, fldr, exe = exe_Jack_Curved):
	start = timer()

	cmd = getSimcmd(simArgument,exe,isPC=True)

	print("Foldername :", fldr)
	# Commands to create folder and run jobs

	os.system("mkdir "+ fldr )
	os.chdir(fldr)
	os.system(cmd)
	end = timer()

	#recording the simulation time in a usable format
	logerr = open("log.err","w")
	time = end-start
	m = int(time//60)
	s = time%60
	logerr.write(f"real\t{m}m{s:.3f}s")
	logerr.close()
	
	dumpDictionaryJSON(simArgument,"simArgument")
	dumpDictionaryJSON(params,"params")

	os.chdir("..")
	#print(f"Simulation length: {end-start} seconds.")
	return

#a few good examples of param and sim dictionaries that drivers commonly use
example_params = {#solution characteristics
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

example_simArgument = {#arguments for a simulation
		'nsweeps': 1000,
		'nsnap': 10,
		'npart': 100,
		'radius': 100,            #[2a]
		'temp': 1,               #[kT]
		'rc': 5,        #[2a]
		'length_scale': 1e-2,           #[2a]
		'fieldk': 1,                   #[kT/(2a)^2]
		'a': 6000,                      #[kT]
		'i': 1,						# 1 denotes yukawa interactions in the arclength coordinate
		'start_from_config': False,
		}
