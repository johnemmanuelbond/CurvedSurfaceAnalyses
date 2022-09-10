# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 3, 2022

Used to categorize a batch of MC results, extract relevant infromation for
the reader, compute density profiles, compare them to theory, make movies,
and examine potential order parameters

@author: Jack Bond
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os, glob, sys
import json, re
from timeit import default_timer as timer

from datetime import datetime

import UnitConversions as units
import ForceBalanceTheory as model
import OrderParameters as order
import DriverUtilities as util
import FileHandling as handle

"""
chops the first N particles (usually also the top N) out from a frame and saves it
as a new file.
"""
def chopCap(frame, newN, name = "N_n_R_r_V_v"):
	top = frame[np.argsort(frame[:,2])][-newN:]
	handle.save_xyz(top,f"{os.getcwd()}/{name}.xyz")	

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
The Monte carlo spits out a configuration file which has 3 parts:
simargument: contains parameters in simulation units necessary to run a sim
interactions: contains a list of interparticle potentials with energy scales in kT units
params: contains SI parameters necessary for translating simulation results to experimnental results
By reading these config files we can group simulation folders into groups
to which we can apply the exact same analysis, or otherwise match up these
lists of parameters to the appropriate simulation folders.
"""
def categorize_batch():

	# we want to log any calculations we do:
	log = open('log.txt','a')
	# datetime object containing current date and time
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	#Monte Carlo Outputs always contain a dictionary called configFile.json
	#We can use this file to find all the folders which contain MC data
	sims = np.array(glob.glob("*/configFile.json"))
	configs = []
	for i,sim in enumerate(sims):
		config = json.load(open(sim,'r'))
		sims[i] = sim[:-15]
		configs.append(config)

	#if the sim argument and interactions are the same
	#accross two sims, they are considered two "seeds" of the
	#same simulation and can be grouped
	start = timer()
	isSame = np.zeros((len(configs),len(configs)))
	for i, c1 in enumerate(configs):
		for j, c2 in enumerate(configs):
			if j < i:
				pass
			same = 1
			d1,d2 = c1['simargument'],c2['simargument']
			for arg in d1:
				if arg!="i" and arg!="a" and arg!="length_scale":
					if d1[arg] != d2[arg]:
						same = 0
			d1,d2 = c1['interactions'][0],c2['interactions'][0]
			for arg in d1:
				if d1[arg] != d2[arg]:
					same = 0
			isSame[i,j] = same
			isSame[j,i] = same
	nseeds = int(np.sum(isSame,axis=0)[0])
	ndiff = isSame[0].size//nseeds
	end = timer()
	pairs = np.array(np.unique(isSame,axis=0))
	log.write(f"{dt}:: config Dictionary Comparison Time: {end-start}s\n")


	#once we've determined the grouping in the isSame array, we simply assign
	#a config dictionary from the seeds and select the appropriate folder
	#names from the list we globbed
	start = timer()
	seedFolders = []
	configs_2 = []
	for i,l in enumerate(pairs):
		seedFolders.append(sims[l==1])
		configs_2.append(configs[np.where(l==1)[0][0]])
	end = timer()
	log.write(f"{dt}:: simArgument dictionary, params dictionary, and seed directory list organization time: {end-start}s\n")
	
	log.write("\n")
	log.close()
	return configs_2, seedFolders


"""
SOON TO BE DEPRECATED
Old Versions of the monte carlo used to spit out 

The driver file outputs a json of the simulation argument in each seed folder.
The driver also outputs a json of the experimental parameters, though they sometimes
change between sims. Using these jsons we can organize the directory into lists of folders
where the simulation parameters are the same and the only difference is the initial state.
For each of these lists of folders, we also want a python dictionary of the simulation
argument and a python dictionary of the experimental parameters.
"""
def categorize_batch_old():

	# we want to log any calculations we do:
	log = open('log.txt','a')
	# datetime object containing current date and time
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	#simulation output files are always in folders including the word "snapshots",
	#so we locate all these folders and get the jsons from each
	sims = np.array(glob.glob("*snapshots*/"))
	simDicts = []
	for sim in sims:
		simArgument = json.load(open(sim + "simArgument.json",'r'))
		simArgument['old'] = True
		if(os.path.exists(sim + "configFile.json")):
			simArgument['old'] = False
		simDicts.append(simArgument)

	#if the simArgument dictionary is the same accross two sims, they are considered
	#two "seeds" of the same simulation and can be grouped
	start = timer()
	isSame = np.zeros((len(simDicts),len(simDicts)))
	for i, d1 in enumerate(simDicts):
		for j, d2 in enumerate(simDicts):
			if j < i:
				pass
			same = 1
			for arg in d1:
				if arg!="i":
					if d1[arg] != d2[arg]:
						same = 0
			isSame[i,j] = same
			isSame[j,i] = same
	nseeds = int(np.sum(isSame,axis=0)[0])
	ndiff = isSame[0].size//nseeds
	end = timer()
	pairs = np.array(np.unique(isSame,axis=0))
	log.write(f"{dt}:: simArgument Dictionary Comparison Time: {end-start}s\n")


	#once we've determined the grouping in the isSame array, we simply assign a
	#simArgument dictionary from the seeds, slightly modify the experimetal parameter
	#dictionary to reflect the changing conditions, and select the appropriate folder
	#names from the list we globbed
	start = timer()
	seedFolders = []
	paramDicts = []
	simDicts_2 = []
	for i,l in enumerate(pairs):
		seedFolders.append(sims[l==1])
		simDicts_2.append(simDicts[np.where(l==1)[0][0]])
		params = json.load(open(seedFolders[i][0]+"params.json", 'r'))
		paramDicts.append(params)
	end = timer()
	log.write(f"{dt}:: simArgument dictionary, params dictionary, and seed directory list organization time: {end-start}s\n")
	
	log.write("\n")
	log.close()
	return simDicts_2, paramDicts, seedFolders

"""
Marcc spits out a log.err file, hidden within which is the time it took
the supercomputer to perform the simulation job. This method finds that time
and reports the mean time for a batch of similar seeds.
"""
def get_average_computation_time(seedFolders, label = "N_n_R_r_V_v"):
	# we want to log any calculations we do:
	log = open('log.txt','a')
	# datetime object containing current date and time
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	#pattern for where the computation time is in the log.err file
	pattern = re.compile(r"real\s\d+m\d+\.\d+s")

	#go through that file in each seed and convert that string to a time in s
	times = []
	for seed in seedFolders:
		logerr = open(seed+"log.err","r")
		found = False
		for line in logerr:
			for match in re.finditer(pattern,line):
				if not found:
					minutes = int(str(re.findall(r"\d+m", line))[2:-3])
					seconds = int(str(re.findall(r"\d+\.", line))[2:-3])
					millis = int(str(re.findall(r"\d+s", line))[2:-3])
					times.append(60*minutes+seconds+millis/1000)
					found = True
	mean_t = np.mean(times)

	log.write(f"{dt}:: {label} mean computation time: {mean_t}\n\n")
	log.close()
	return mean_t, np.array(times)

"""
Once we've categorized data into groups of seeds, we can read all such like data together.
we usually want to label this data with something indicative of the independent variables
we change. last_section refers to how many of the snapshots we want to consider, equilibration
happens at some point early, though the sim considers more completely random moves throughout
the first half, so I default to only considering the last third of snapshots. This method
saves the resultant trajectory to a file for faster reading in the future, reset is used
to bypass this.
"""
def sample_frames(seedFolders, label = "N_n_R_r_V_v", last_section = 1/3, reset = False):
	
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()

	#if the npy file exists we can just load that
	preloaded = os.path.exists(f"{label}_traj_xyz.npy")
	if ((not reset) and preloaded):
		multiple = np.load(f"{label}_traj_xyz.npy")
		log.write(f"{dt}:: Re-read {label} monte carlo files\n")
	#otherwise we have to read each file individually and group them into a single array
	else:
		tmp = []
		hasOutput0 = 1.0*os.path.exists(seedFolders[0]+"output_0.xyz")
		maxlabel = len(glob.glob(seedFolders[0]+"output*.xyz"))-hasOutput0
		for seed in seedFolders:
			if (last_section<0):
				if(last_section%1!=0): raise Exception("Noninteger negative last_section")
				tmp.append(handle.read_xyz_frame(seed+"output_"+str(int(maxlabel+last_section))+".xyz"))
			else:
				if(last_section>1): raise Exception("Nonfraction positive last_section")
				for l in np.arange(int((1-last_section)*maxlabel)+1-hasOutput0,maxlabel):
					tmp.append(handle.read_xyz_frame(seed+"output_"+str(int(l))+".xyz"))
		multiple = np.stack(tmp).squeeze()
		np.save(f"{label}_traj_xyz.npy", multiple)

		end = timer()
		log.write(f"{dt}:: Read {label} monte carlo files in {end-start}s\n")

	log.write("\n")
	log.close()
	return multiple

"""
Given a batch of seeds and the respective dictionaries--one for the experimental parameters,
one for the simulation argument--this method will return binned density and area fractions
as a function of arclength. Additionally it will use the osmotic force balance theory
to predict the density profile that the experimental parameters would produce.
"""
def compute_SI_density_profile(seedFolders, config, label = "N_n_R_r_V_v", bs=1):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label))

	simarg = config['simargument']
	params = config['params']
	inter = config['interactions'][0]

	#relevant lengths in relevant units
	a = params['particle_radius']*1e6 #microns
	aeff = units.getAEff(params)*1e6 #microns
	R = simarg['radius'] #2a

	N = simarg['npart']

	#get theoretical area fraction
	eta_th, rs, guess_N = model.profileFromFieldStrength(params,N, shellRadius = R, tol_e=-3)
	rs*=2*a #convert to mircons

	#mids, rho, rho_bin = rho_hist(frames, bin_width=bs)
	mids, rho, rho_bin = spherical_rho_hist(frames, R, bin_width=bs)
	log.write(f"{dt}:: Histogram bin size: {bs}\n")
	
	#check particle number against theory
	target_N = integrate_histogram(rho,rho_bin, shellRadius=R)
	log.write(f"{dt}:: Check {label} particle number consistency:\n  Target N: {N}\t Integrated Profile: {target_N:.5f}\t Integrated Theoretical Profile: {guess_N:.5f}\n")

	mids*=2*a #convert to microns
	rho*=(1/(2*a))**2 #convert to per micron^2
	eta = np.array([units.eta(r, aeff) for r in rho]) #get the area fraction from the density

	end = timer()
	log.write(f"{dt}:: Computed {label} density profiles in {end-start}s\n\n")
	log.close()

	return mids, rho, eta, rs, eta_th

"""
calculates particle number density as a function of arclength on a sphere
given a N x M x d array of coordinates and an N x M array of charges, as
as well as that sphere's radius--which is necessary for computing arclengths.
"""
def spherical_charge_hist(frames, charges, shellRadius, furthest=None, bin_width=2):
	fnum, pnum, _ = frames.shape #get number of frames and particles
	
	zcoords = frames[:,:,-1].flatten()
	charges = charges.flatten()
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
	print(hbin_edge)

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2
	
	#so I did this calculus on paper, it would be helpful to find a reference for it.
	area = (2*np.pi*shellRadius**2)*(np.cos(hbin_edge[:-1]/shellRadius)-np.cos(hbin_edge[1:]/shellRadius))
	hval = 0*mids

	for i, l in enumerate(mids):
		relevant = 1*(arclengths>=hbin_edge[i])*(arclengths<=hbin_edge[i+1])
		print(relevant.size)
		hval[i] = np.sum(charges[relevant==1])
		print(hval[i]/fnum)

	#hval, hbin = np.histogram(arclengths, bins=hbin_edge)
	
	charge = hval / (fnum*area)
	
	return mids, charge, hbin_edge

"""
Given a batch of seeds and the respective dictionaries--one for the experimental parameters,
one for the simulation argument--this method will return binned charge as a function of arclength.
"""
def compute_topological_charge_profile(seedFolders, config, label = "N_n_R_r_V_v", bs=1):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label))

	simarg = config['simargument']
	params = config['params']
	inter = config['interactions'][0]

	#relevant lengths in relevant units
	a = params['particle_radius']*1e6 #microns
	aeff = units.getAEff(params)*1e6 #microns
	R = simarg['radius'] #2a

	N = simarg['npart']

	charges = [];
	for frame in frames:
		q = 6-order.Vc(frame,R=R)
		charges.append(q)
	charges = np.array(charges)

	mids, charge, charge_bin = spherical_charge_hist(frames, charges, R, bin_width=bs)

	log.write(f"{dt}:: Histogram bin size: {bs}\n")
	avg_charge = integrate_histogram(charge, charge_bin, shellRadius=R)
	log.write(f"{dt}:: Found average topological charge: {avg_charge}\n")

	mids*=2*a #convert to microns
	charge*=(1/(2*a))**2 #convert to per micron^2

	end = timer()
	log.write(f"{dt}:: Computed {label} density profiles in {end-start}s\n\n")
	log.close()

	return mids, charge, avg_charge

def pair_charge_correlation(q1,q2,frame, shellRadius, bin_width=2):	
	qs = 6-order.Vc(frame,R=shellRadius)
	#qs = 6-order.Nc(frame,shellradius=order.firstCoordinationShell(frame))

	hbin_edge = np.histogram_bin_edges(range(10),
							   bins=int(np.pi*shellRadius/bin_width),
							   range=(0, np.pi*shellRadius))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2
	hval = 0*mids

	q1s = frame[qs==q1]
	Nq1 = len(q1s)
	q2s = frame[qs==q2]
	Nq2 = len(q2s)
	
	norm = Nq1*Nq2-Nq1*(q1==q2)

	thetas = np.einsum("ik,i,jk,j->ij",q1s,1/np.linalg.norm(q1s,axis=-1),q2s,1/np.linalg.norm(q2s,axis=-1))
	thetas[thetas>1.0]=1.0
	thetas[thetas<-1.0]=-1.0

	ws = shellRadius*np.arccos(thetas).flatten()		
	counts,_ = np.histogram(ws, bins=hbin_edge)
	hval = counts/(norm/2*(np.cos((mids-widths/2)/shellRadius)-np.cos((mids+widths/2)/shellRadius)))

	return mids, hval, hbin_edge, qs

def compute_average_pair_charge_correlation(q1,q2,seedFolders, simArgument, label = "N_n_R_r_V_v", bs = np.pi/40):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label,last_section=-1))

	#relevant lengths in relevant units
	R = simArgument['radius'] #2a
	Qs = []
	hvals = []
	midss = []

	for frame in frames:
		#qs = 6-order.Nc(frame,shellradius=order.firstCoordinationShell(frame))
		#print(np.sum(qs),qs[qs==1].size,qs[qs==-1].size)
		mids, hval, _, qs = pair_charge_correlation(q1,q2,frame,R,bin_width=R*bs)
		midss.append(mids)
		hvals.append(hval)
		Qs.append(np.sum(qs))
	
	Qs = np.array(Qs)
	mids = np.mean(np.array(midss),axis=0)
	g = np.mean(np.array(hvals),axis=0)


	log.write(f"{dt}:: Histogram bin size: {bs:.3f}\n")
	avg_charge = np.mean(Qs)
	log.write(f"{dt}:: Found average topological charge: {avg_charge}\n")

	mids*=1/R #convert to angles

	end = timer()
	log.write(f"{dt}:: Computed {label} pair correlation functions in {end-start}s\n\n")
	log.close()

	return mids, g, Qs

from scipy.signal import find_peaks
def firstCoordinationShell(seedFolders,label):
	
	frames = np.array(sample_frames(seedFolders,label=label,last_section=1/2))
	npart = frames[0].shape[0]

	dists = []
	frameshells = []
	for f in frames:
		_,info = order.radialDistributionFunction(f)
		dists.append(info["distance_list"])
		frameshells.append(info["first_coordination_shell"])

	dists = np.array(dists).flatten()
	hbin_edge = np.histogram_bin_edges(dists,
								   bins=int(npart/10),
								   range=(0, 3*min(dists)))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2

	hval, _ = np.histogram(dists, bins=hbin_edge)

	peaks, _ = find_peaks(hval,prominence=10)

	# if(len(peaks)<2):
	# 	print(label)
	# 	print(mids[peaks])
	# 	raise Exception("Not enough peaks")

	# shellRadius = (mids[peaks[1]]+mids[peaks[0]])/2
	spacing = mids[peaks[0]]

	relevant = (mids>spacing)*(mids<2*spacing)
	relevantMids = mids[relevant]
	relevantHval = hval[relevant]
	shellRadius = relevantMids[np.argmin(relevantHval)]

	return mids, hval, shellRadius, spacing

def scar_correlation(frame, shellRadius, bin_width=2):	
	scars, scarCharges = order.findScars(frame)

	hbin_edge = np.histogram_bin_edges(range(10),
							   bins=int(np.pi*shellRadius/bin_width),
							   range=(0, np.pi*shellRadius))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2
	hval = 0*mids

	#print(scars, np.where(scarCharges!=0)[0])
	chargedScars = [scars[i] for i in np.where(scarCharges!=0)[0]]

	meanscarpositions = np.array([np.mean(frame[scar],axis=0) for scar in chargedScars])
	meanscarpositions = np.array([shellRadius*p/np.linalg.norm(p,axis=-1) for p in meanscarpositions])
	
	Nscar = len(meanscarpositions)

	norm = Nscar*(Nscar-1)

	thetas = np.einsum("ik,i,jk,j->ij",meanscarpositions,1/np.linalg.norm(meanscarpositions,axis=-1),meanscarpositions,1/np.linalg.norm(meanscarpositions,axis=-1))
	thetas[thetas>1.0]=1.0
	thetas[thetas<-1.0]=-1.0

	ws = shellRadius*np.arccos(thetas).flatten()		
	counts,_ = np.histogram(ws, bins=hbin_edge)
	hval = counts/(norm/2*(np.cos((mids-widths/2)/shellRadius)-np.cos((mids+widths/2)/shellRadius)))

	return mids, hval, scars, meanscarpositions

def compute_average_scar_correlation(seedFolders, config, label = "N_n_R_r_V_v", bs = np.pi/40):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label,last_section=-1))

	#relevant lengths in relevant units
	R = config['simargument']['radius'] #2a
	scarCount=[]
	hvals = []
	midss = []

	for frame in frames:
		#qs = 6-order.Nc(frame,shellradius=order.firstCoordinationShell(frame))
		#print(np.sum(qs),qs[qs==1].size,qs[qs==-1].size)
		mids, hval, _,meanscarpositions = scar_correlation(frame,R,bin_width=R*bs)
		midss.append(mids)
		hvals.append(hval)
		scarCount.append(meanscarpositions.shape[0])
	
	mids = np.mean(np.array(midss),axis=0)
	g = np.mean(np.array(hvals),axis=0)


	log.write(f"{dt}:: Histogram bin size: {bs:.3f}\n")
	avg_scarCount = np.mean(scarCount)
	log.write(f"{dt}:: Found average scar count: {avg_scarCount}\n")

	mids*=1/R #convert to angles

	end = timer()
	log.write(f"{dt}:: Computed {label} scar correlation functions in {end-start}s\n\n")
	log.close()

	return mids, g, scarCount

#given a histogram of something, takes the negative log of the probability
#distribution in order to get an 'energy'
def BoltzmannInversion(mids,widths,hval):
	norm = np.sum(widths*hval)
	Ukt = -1*np.log(hval/norm)
	return mids, Ukt


if __name__=="__main__":

	start = timer()

	configs, seedFoldersList = categorize_batch()

#Radial Distribution Function
	figRad, axRad = plt.subplots()
	axRad.set_title("Radial DistributionFunctions for Long-Range Repulsion")
	axRad.set_xlabel(r"Euclidean Displacement [$2a$]")
	axRad.set_ylabel("Counts")

#Excess Charge v Sweeps Visualization
	figCharge, axCharge = plt.subplots()
	axCharge.set_title("Excess Charge vs Sweeps")
	axCharge.set_ylabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axCharge.set_xlabel("sweeps")

#Excess Charge v system variables Visualization
	figXS1, axXS1 = plt.subplots()
	axXS1.set_title("Excess Charge vs Initial R/a")
	axXS1.set_ylabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axXS1.set_xlabel("R/a")

	figXS2, axXS2 = plt.subplots()
	axXS2.set_title(r"Excess Charge vs $\eta_{{eff}}$")
	axXS2.set_ylabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
	axXS2.set_xlabel(r"$\eta_{{eff}}$")

	avg_excess_charge = []
	R_on_a = []
	etas = []

	for i,seedFolders in enumerate(seedFoldersList):
		simarg = configs[i]['simargument']
		params = configs[i]['params']
		N = simarg['npart']
		R = simarg['radius']
		a = params["particle_radius"]
		aeff = units.getAEff(params)
		eta_eff = N*(aeff/(2*a))**2/(4*R**2)

		initFrame = handle.read_xyz_frame(seedFolders[0]+"output_0.xyz")
		_,info = order.radialDistributionFunction(initFrame)
		spacing = info['particle_spacing']

		lab = f"eta_eff={eta_eff:.3f},R={R:.1f}"
		midsRad, hval, coordinationShell, _ = firstCoordinationShell(seedFolders, label=lab)
		Ra = R/spacing
		
		pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R/a={Ra:.1f}"

		frames = np.array(sample_frames(seedFolders,label=lab,reset=True))

		vcs = [order.Vc(frame, R = R) for frame in frames]
		XS = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)

		avg_excess_charge.append(np.mean(XS))
		R_on_a.append(Ra)
		etas.append(eta_eff)

		#use to filter what shows up in the plot, for example we can do eta_eff>0.71
		cond = True

		#use to decide which seed we will plot against sweeps
		numSim=40

		if(cond):

			#print(f"{pltlab} first coordination shell: {coordinationShell} [2a]")
			axRad.plot(midsRad,hval, label = pltlab, lw=0.6)
			sim = np.array(sample_frames(seedFolders[numSim:numSim+1], label = lab+f"_{numSim}th_seed", last_section = 1.0))
			
			sweeps = simarg['nsnap']*(1+np.arange(sim.shape[0]))
			#ncs = [order.Nc(frame, shellradius=coordinationShell) for frame in sim]
			vcs = [order.Vc(frame) for frame in sim]

			excessVC = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)
			axCharge.plot(sweeps,excessVC,label=pltlab)#, c = cm.hsv(i/len(seedFoldersList)),lw=0.6)

	avg_excess_charge = np.array(avg_excess_charge)
	R_on_a = np.array(R_on_a)
	etas = np.array(etas)

	axRad.legend()
	figRad.savefig(f"Radial Distribution Function_{section}.png")

	#plots XS charge vs R/a for each eta
	for h in np.unique(etas):
		xs = R_on_a[etas==h]
		ind = np.argsort(xs)
		ys = avg_excess_charge[etas==h]
		axXS1.plot(xs[ind],ys[ind],label=rf"$\eta_{{eff}}$={h}",lw=0.6)
	axXS1.legend()
	figXS1.savefig("Excess Charge vs R over a.png")

	#plots XS charge vs eta for each R/a
	for r in np.unique(R_on_a):
		xs = etas[R_on_a==r]
		ind = np.argsort(xs)
		ys = avg_excess_charge[R_on_a==r]
		axXS2.plot(xs[ind],ys[ind],label=rf"R/a={r:.2f}",lw=0.6)
	axXS2.legend()
	figXS2.savefig("Excess Charge vs effective Eta.png")

	axCharge.legend()
	figCharge.savefig(f"{numSim}th seed -- Excess Charge per Sweep.png")

	sweeps = configs[0]['simargument']['nsweeps']
	axCharge.set_xlim([sweeps/2,sweeps])
	axCharge.set_ylim([0,10])
	figCharge.savefig(f"{numSim}th seed -- Excess Charge per Sweep (last half).png")


	end = timer()
	print(f"{end-start}s runtime")


















# def var_bin_rho_hist(frames, var_bins=None):
#     """calculates particle number density projected onto xy-plane
#     given a N x M x 3 array. Uses flexible bin size, starting at a_hc and
#     decreasing to a_hc*1e-2 at furthest extent."""
#     projected = frames[:,:,:2]
#     fnum, pnum, _ = projected.shape #get number of frames and particles

#     # converts projected coords into distances from center across sample
#     dists = np.linalg.norm(projected, axis=-1).flatten()

#     # if furthest is not defined, include 20% beyond farthest excursion
#     if var_bins is None:
#         bin_widths = 0.1*np.ones(50)
#         var_widths = np.geomspace(a_hc, 0.1, num=40)
#         bin_widths[:var_widths.size] = var_widths
#         var_bins = np.zeros(bin_widths.size+1)
#         var_bins[1:] = np.cumsum(bin_widths)

#     widths = var_bins[1:] - var_bins[:-1]
#     mids = var_bins[:-1] + widths/2

#     #annular area formula as defined below:
#     #https://en.wikipedia.org/wiki/Annulus_(mathematics)
#     area = np.pi*(var_bins[1:]**2 - var_bins[:-1]**2)
#     #get count within each bin
#     hval, hbin = np.histogram(dists, bins=var_bins)
	
#     errs = np.sqrt(hval) / (fnum * area)
#     rho = hval / (fnum * area)
#     return mids, rho, hbin, errs