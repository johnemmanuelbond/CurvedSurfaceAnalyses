import numpy as np
import random

import os, glob
import json, yaml
from timeit import default_timer as timer

from datetime import datetime

import FileHandling as handle




""" FOR RUNNING MONTE CARLO SIMS """

#Flat-Case executables
exe_Marcc_Flat = "~/bin/plane_yukawa_shell"
exe_Jack_Flat = "C:/Users/johne/MolecularMonteCarlo/FlatSourceCode/plane_yukawa_field/build/bin/plane_yukawa_shell.exe"

#Spherical Shell executables
exe_Marcc_Curved = "~/bin/yukawa_shell"
exe_Jack_Curved =  "C:/Users/johne/MolecularMonteCarlo/MCEngine/shell_yukawa_field/build/bin/yukawa_shell.exe "

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

def getRandomInputFile(dicString):
	files = glob.glob(dicString)
	return random.choice(files)

#runs a simulation using Marcc's parallelism
def runMarccSimWithJSON(config, fldr, exe = exe_Marcc_Curved):
	pwd = os.getcwd()
	print("Foldername :", fldr)
	# Commands to create folder and run jobs

	os.system("mkdir "+fldr)

	os.system("cp sub.sh sub_edit.sh")

	fin = open("sub_edit.sh", "rt")
	data = fin.read()
	data = data.replace('xxx',f"{pwd}/{fldr}")
	data = data.replace('yyy',"time "+exe)
	fin.close()

	fin = open("sub_edit.sh", "wt")
	fin.write(data)
	fin.close()

	#os.system('echo "'+cmd+'" >> sub_edit.sh')
	os.system("mv sub_edit.sh " + fldr + "/sub.sh")

	os.chdir(fldr)

	handle. dumpDictionaryJSON(config,"configFile")

	os.system("sbatch sub.sh;")

	os.chdir("..")

	return

#runs a simulation using a version of the code compiled on a windows machine
def runPCSimWithJSON(config, fldr, exe = exe_Jack_Curved):
	start = timer()

	print("Foldername :", fldr)
	# Commands to create folder and run jobs

	os.system("mkdir "+ fldr )
	os.chdir(fldr)

	handle.dumpDictionaryJSON(config,"configFile")

	os.system(exe)
	end = timer()

	#recording the simulation time in a usable format
	logerr = open("log.err","w")
	time = end-start
	m = int(time//60)
	s = time%60
	logerr.write(f"real\t{m}m{s:.3f}s")
	logerr.close()

	os.chdir("..")
	#print(f"Simulation length: {end-start} seconds.")
	return

"""
sub.sh
	
	#!/bin/bash

	#SBATCH --job-name=ss_shell
	#SBATCH --time=72:0:0
	#SBATCH --partition=defq
	#SBATCH --chdir=xxx
	#SBATCH --error=log.err
	#SBATCH --account=mbevan4

	#### load and unload modules you may need
	module load gcc
	module load armadillo
	module load openblas
	module load anaconda
	module load hdf5
	module list

	echo "Job $SLURM_JOBID initiated"

	yyy;

	time python ~/code/CurvedSurfaceAnalyses/visualize.py
"""

params = {#solution characteristics
		'temperature': 298,             # [K]
		'rel_permittivity': 78,         # unitless
		'ion_multiplicity': 1,       # unitless
		'debye_length': 10e-9,          	# [m]
		#particle characteristics
		'particle_radius': 1.4e-6,      # [m]
		'surface_potential': -75e-3,    # [V]
		'Gamma': 0.0,
		#'surface_potential': -75e-3,   # [V]
		'fcm':  -0.2287,                # unitless
		#field characteristics
		'vpp': 0.0,                     # [V]
		'dg': 100e-6,                   # [m]
		}

interactions = [
			{"key": "exp",
			"A": 6050,
			"p": 0.002,
			}
			]

simargument = {#arguments for a simulation
		'nsweeps': 50000,
		'nsnap': 50,
		'npart': 92,
		'radius': 5.0,            	#[2a]
		'temp': 1.0,               	#[kT]
		'rc': 5.0,        			#[2a]
		'fieldk': 0.0,				#[kT/(2a)^2]
		'rand_move_frac': 0.0,
		'datapath': "?",
		}

config = {
	'simargument': simargument,
	'interactions': interactions,
	'params':params,
}




""" OLD LYRA-ORIENTED METHODS FOR REFERENCE """

#turns a dictionary of the MC code's lyra keys into a string
#which the command line can run
def getSimcmd(simArgument, exepath, isPC = False):
	cmd = "time " + exepath
	if isPC:
		cmd = exepath

	for arg in simArgument:
		if arg == "a" or arg == "i" or arg == 'k':
			cmd += f" -{arg} {simArgument[arg]}"
		else:
			cmd += f" --{arg} {simArgument[arg]}"

	print(cmd)
	return cmd

#runs a simulation using Marcc's parallelism
def runMarccSimWithLyra(simArgument, params, fldr, exe = exe_Marcc_Curved):
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

	handle.dumpDictionaryJSON(simArgument,"simArgument")
	handle.dumpDictionaryJSON(params,"params")

	os.chdir("..")

	return

#runs a simulation using a version of the code compiled on a windows machine
def runPCSimWithLyra(simArgument, params, fldr, exe = exe_Jack_Curved):
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
	
	handle.dumpDictionaryJSON(simArgument,"simArgument")
	handle.dumpDictionaryJSON(params,"params")

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
		}




""" FOR RUNNING LAMMPS SIMS """

def runLammpsSimMarcc(config, fldr, inputStructPath, inFile = 'diff_field.in'):
	os.system("mkdir "+fldr)
            
    os.system(f"cp {inFile} edit.in")
    for key in config:
    	os.system(f"sed -i 's/{key}/{config[key]:0.5f}/' edit.in")
    os.system(f"mv edit.in {fldr}/{inFile}")
    os.system(f"cp {inputStructPath} {fldr}/input.data")

    os.chdir(fldr)
    os.system("sbatch ~/bin/subLAMMPS.sh")
    os.chdir("..")

#example of the kinds of keys and values that could appear in diff_field.in
config = {#Arguments for running a lammps sim
	'xxxradiusxxx': 8.0, #[2a]
	'xxxnsnapxxx': 2000,
	'xxxmassxxx': 1.0,
	'xxxdampxxx': 1.0, #[tau]
	'xxxtimestepxxx': 1e-4, #[s]
	'xxxnstepxxx': 90000000,
	'xxxtempxxx': 1.0, #[kT]
}

"""
must have lammps compiled with the manifold package and this bash script in your bin

subLammps.sh

	#!/bin/bash

	#SBATCH --job-name=lmp_coll
	#SBATCH --time=72:00:00
	#SBATCH --error=log.err
	#SBATCH --partition=defq
	#SBATCH --account=mbevan4


	#### load and unload modules you may need
	module list

	echo "Job $SLURM_JOBID initiated"

	lmp -in *.in;
	python ~/bin/general_analysis.py

"""