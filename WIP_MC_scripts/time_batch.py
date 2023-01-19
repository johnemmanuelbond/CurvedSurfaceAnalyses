# -*- coding: utf-8 -*-
"""
Created on Sun, Jun 26, 2022

uses MCBatchAnalyzer to plot computation cost of various sims.

@author: Jack Bond
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from MCBatchAnalyzer import categorize_batch_old,get_average_computation_time

simDicts, paramDicts, seedFoldersList = categorize_batch_old()

times = []
Ns = []
ps = []
olds = []

for i, seeds in enumerate(seedFoldersList):
	olds.append(simDicts[i]['old'])
	Ns.append(simDicts[i]['npart'])
	ps.append(simDicts[i]['length_scale'])
	
	if(simDicts[i]['old']):
		lab = f"old;N={simDicts[i]['npart']}_p={simDicts[i]['length_scale']:.3f}"
	else:
		lab = f"new;N={simDicts[i]['npart']}_p={simDicts[i]['length_scale']:.3f}"

	times.append(get_average_computation_time(seeds,label = lab)[0])

times = np.array(times)
Ns = np.array(Ns)
ps = np.array(ps)
olds = np.array(olds)

fig,ax = plt.subplots()
ax.set_title("Computation time vs System Size")
ax.set_ylabel("time [s]")
ax.set_xlabel("N")
for p in np.unique(ps[olds]):
	xs = Ns[olds][ps[olds] == p]
	ind = np.argsort(xs)
	ys = times[olds][ps[olds] == p]
	ax.plot(xs[ind],ys[ind], label = f"OLD: yukawa, {p:.4f}")
news = np.invert(olds)
for p in np.unique(ps[news]):
	xs = Ns[news][ps[news] == p]
	ind = np.argsort(xs)
	ys = times[news][ps[news] == p]
	ax.plot(xs[ind],ys[ind], label = f"NEW: yukawa, {p:.4f}")
ax.legend()
fig.savefig("timeComparison")
