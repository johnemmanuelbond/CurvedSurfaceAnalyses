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

"""Reads xyz coordinates from a .xyz file. Expected format:
number of particles
comment line (should have units)
type x-coord y-coord z-coord
type x-coord y-coord z-coord

Returns a triply nested numpy array, with format:
[ # first frame
 [-10.34, -10.8, 37.1], #coordinates of particle
 [-14.48, 5.69, 36.85],
 [-7.47, -16.2, 35.8],
"""
def read_xyz_frame(filename):
	frame = []
	with open(filename, 'r') as xyzfile:
		pnum = None
		for i, line in enumerate(xyzfile):
			if i == 0:
				pnum = float(line.strip())
			elif i == 1:
				pass  #throw away comment
			elif i <= pnum+1:
				# assumes format as below for particle coordinates
				# index xcomp ycomp zcomp ... (unspecified afterwards)
				coordinates = line.split()[1:4]
				frame.append([float(coord) for coord in coordinates])
			else:
				print("extra: " + line)
	return np.array(frame)

"""Given a frame or set of frames, saves them to filename as xyz file"""
def save_xyz(coords, filename, comment=None):
    if comment == None:
        comment = "idx x(um)   y(um)   z(um)   token\n"
    if len(coords.shape) == 2:
        coords = coords[np.newaxis,:] #make single frames correct size
    print(filename)
    with open(filename, 'w', newline='') as output:        
        for i, frame in enumerate(coords):
            #print number of particles in frame
            output.write("{}\n".format(frame.shape[0]))
            output.write(comment)
            for j, part in enumerate(frame):
                output.write("C {:.6e} {:.6e} {:.6e} \n".format(
                             *part))

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
The driver file outputs a json of the simulation argument in each seed folder.
The driver also outputs a json of the experimental parameters, though they sometimes
change between sims. Using these jsons we can organize the directory into lists of folders
where the simulation parameters are the same and the only difference is the initial state.
For each of these lists of folders, we also want a python dictionary of the simulation
argument and a python dictionary of the experimental parameters.
"""
def categorize_batch():

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
		for line in logerr:
			for match in re.finditer(pattern,line):
				minutes = int(str(re.findall(r"\d+m", line))[2:-3])
				seconds = int(str(re.findall(r"\d+\.", line))[2:-3])
				millis = int(str(re.findall(r"\d+s", line))[2:-3])
				times.append(60*minutes+seconds+millis/1000)
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
		maxlabel = len(glob.glob(seedFolders[0]+"output*"))
		for seed in seedFolders:
			for l in np.arange(int((1-last_section)*maxlabel),maxlabel):
				tmp.append(read_xyz_frame(seed+"output_"+str(l)+".xyz"))
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
def compute_SI_density_profile(seedFolders, params, simArgument, label = "N_n_R_r_V_v", bs=1):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label))

	#relevant lengths in relevant units
	a = params['particle_radius']*1e6 #microns
	aeff = units.getAEff(params)*1e6 #microns
	R = simArgument['radius'] #2a

	N = simArgument['npart']

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
def compute_topological_charge_profile(seedFolders, params, simArgument, label = "N_n_R_r_V_v", bs=1):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label))

	#relevant lengths in relevant units
	a = params['particle_radius']*1e6 #microns
	aeff = units.getAEff(params)*1e6 #microns
	R = simArgument['radius'] #2a

	N = simArgument['npart']

	charges = [];
	for frame in frames:
		q = order.charge(frame,R=R)
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

def pair_charge_correlation(q1,q2,frames, charges, shellRadius, bin_width=2):
	fnum, pnum, _ = frames.shape #get number of frames and particles
	V = 4*np.pi*shellRadius**2

	allrs = []
	norms = []
	for f,frame in enumerate(frames):
		q1s = frame[charges[f]==q1]
		Nq1 = len(q1s)
		q2s = frame[charges[f]==q2]
		Nq2 = len(q2s)

		norms.append(Nq1*Nq2-Nq1*(q1==q2))

		rs = np.zeros((Nq1,Nq2))
		for i, r1 in enumerate(q1s):
			for j, r2 in enumerate(q2s):
				if np.all(r1 == r2):
					rs[i,j] == 0
				else:
					rs[i,j] = shellRadius*np.arccos(np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)))
		allrs.append(rs.flatten())

	hbin_edge = np.histogram_bin_edges(allrs[0],
								   bins=int(np.pi*shellRadius/bin_width),
								   range=(0, np.pi*shellRadius))

	widths = hbin_edge[1:] - hbin_edge[:-1]
	mids = hbin_edge[:-1] + widths/2
	hval = 0*mids

	for i, norm in enumerate(np.array(norms)):
		counts,_ = np.histogram(allrs[i], bins=hbin_edge)
		hval += counts/(norm*fnum) * 2/(np.cos((mids-widths/2)/shellRadius)-np.cos((mids+widths/2)/shellRadius))

	return mids, hval, hbin_edge

def compute_average_pair_charge_correlation(q1,q2,seedFolders, simArgument, label = "N_n_R_r_V_v", bs = np.pi/40):
	#we want to log any time we read a file
	log = open("log.txt", "a")
	now = datetime.now()
	dt = now.strftime("%d/%m/%Y %H:%M:%S")

	start = timer()
	frames = np.array(sample_frames(seedFolders,label=label,last_section=1/3))

	#relevant lengths in relevant units
	R = simArgument['radius'] #2a
	charges = [];
	Qs = []
	for frame in frames:
		q = order.charge(frame,R=R)
		charges.append(q)
		Qs.append(np.sum(q))
	
	charges = np.array(charges)
	Qs = np.array(Qs)

	mids, g, _ = pair_charge_correlation(q1,q1,frames,charges,R,bin_width=R*bs)


	log.write(f"{dt}:: Histogram bin size: {bs:.3f}\n")
	avg_charge = np.mean(Qs)
	log.write(f"{dt}:: Found average topological charge: {avg_charge}\n")

	mids*=1/R #convert to angles

	end = timer()
	log.write(f"{dt}:: Computed {label} pair correlation functions in {end-start}s\n\n")
	log.close()

	return mids, g, Qs

if __name__=="__main__":

	simDicts, paramDicts, seedFoldersList, = categorize_batch()

	#print(get_average_computation_time(seedFoldersList[0])[0])

	As = np.array([d['a'] for d in simDicts])


	#setting up density visualization
	fig, ax = plt.subplots()
	ax.set_title(f"Pair Correlation Functions for Long-Range Repulsion")
	ax.set_xlabel(r"Geodesic Distance [rad/$\pi]$")
	ax.set_ylabel(r"$g_{{55}}$")
	ax.set_ylim([0,2])
	ax.set_xlim([0,1])

	for i,seedFolders in enumerate(seedFoldersList):
		lab = f"A={As[i]:.1f}"
		mids, g, Qs = compute_average_pair_charge_correlation(1,1,seedFolders,simDicts[i],label=lab)
		print(f"{lab}: Total Charge per Frame: {Qs}")
		ax.plot(mids/np.pi, g, label=lab,linewidth = 0.5)

	ax.legend()#bbox_to_anchor=(1.5,1))
	
	fig.savefig("5-5 Pair Correlation.jpg", bbox_inches='tight')

	# lastLabel = 134
	# N = 300
	# for i, seedFolders in enumerate(seedFoldersList):
	# 	for j,folder in enumerate(seedFolders):
	# 		frame = read_xyz_frame(f"{folder}/output_{lastLabel}.xyz")
	# 		R = simDicts[i]['radius'] #2a
	# 		_, Q = order.totalCharge(frame,R=R)
	# 		print(Q)
	# 		top = frame[np.argsort(frame[:,2])][-N:]
	# 		save_xyz(top,f"{os.getcwd()}/FinalStates/InitialState_{j}_A_{simDicts[i]['a']}_long.xyz")


	#experimental conditions/assumptions
	#eta_c = 0.85

	# Rs = np.array([d['radius'] for d in simDicts])
	# vs = np.array([p['vpp'] for p in paramDicts])
	# print(Rs,vs)

	# a = paramDicts[0]['particle_radius']*1e6 #microns
	# aeff = units.getAEff(paramDicts[0])*1e6 #microns

	# #setting up density visualization
	# fig, ax = plt.subplots()
	# ax.set_title(f"Charge Density Profiles, Vpp = 2V, Random Starting States")
	# ax.set_xlabel("Arclength [$\mu m$]")
	# ax.set_ylabel(r"Average Topological Charge Density $[\mu m^{-2}]$")

	# fig2, ax2 = plt.subplots()
	# ax2.set_title(f"Comparison to Theoretical Area Fraction, Vpp = 2V, Random Starting States")
	# ax2.set_xlabel("Arclength [$\mu m$]")
	# ax2.set_ylabel(r"$\eta_{eff}$")
	# ax2.set_ylim([0,1])

	# for i,seedFolders in enumerate(seedFoldersList):
	# 	if simDicts[i]['start_from_config'] == False:
	# 		lab = f"R_{Rs[i]:.2f}"
	# 		if simDicts[i]['start_from_config']==False:
	# 			lab+="_random"
	# 		mids, charge, avg_charge = compute_topological_charge_profile(seedFolders,paramDicts[i],simDicts[i],label=lab)
	# 		mids, rho, eta, rs, eta_th = compute_SI_density_profile(seedFolders,paramDicts[i],simDicts[i],label=lab)
	# 		ax2.plot(rs, eta_th,ls='--')
	# 		ax.scatter(mids, charge, label=lab)
	# 		ax2.scatter(mids, eta, label=lab)

	# ax.legend(bbox_to_anchor=(1.5,1))
	# ax2.legend()#bbox_to_anchor=(1.5,1))

	# fig.savefig("ChargeDensity - Random.jpg", bbox_inches='tight')
	# fig2.savefig("AreaFraction - Random.jpg", bbox_inches='tight')


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