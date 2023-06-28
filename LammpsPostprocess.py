# -*- coding: utf-8 -*-
"""
Created on Tue, Jun 27, 2023

Quickly converts a lammps dump file into the common-ground numpy arrays we use most of the time

@author: Jack Bond
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from main_lib.FileHandling import read_infile, read_dump, read_thermo, get_thermo_time

def plot_thermo(header, data, path):
    for i, item in enumerate(header):
        if item.startswith('Time') or item.startswith('Step'):
            continue
        else:
            fig, ax = plt.subplots()
            ax.plot(data[:,0], data[:,i])
            ax.set_xlabel("[$\\tau$]", fontsize=12)
            ax.set_title(item)
            fig.savefig(path+item+".jpg", bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__"
    
    with open('config.json','r') as f:
        config = json.load(f)
        simarg = config['arg']

    coords, steps = read_dump('out.dump')
    steps*= config['arg']['xxxtimestepxxx']
    np.save("datapts.npy",coords)
    np.save("times.npy",times-times[0])

    lammps_params = read_infile(infile[0])
    thermo_header, thermo = read_thermo('log.lammps')
    plot_thermo(thermo_header, thermo, path)
