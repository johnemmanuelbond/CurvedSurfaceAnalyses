# -*- coding: utf-8 -*-
"""
Created on Sat, Jan 1a, 2023

Plots the radial distribution function at several bin widths for a single lammps run

Also performs fits to find the peak heights

@author: Jack Bond, Alex Yeh
"""

import glob, os, json
import numpy as np

from UnitConversions import getAEff, kb
from FileHandling import read_infile, read_dump, read_thermo, get_thermo_time
from OrderParameters import g_r

from numpy.random import default_rng
import matplotlib.pyplot as plt

from timeit import default_timer as timer

from scipy.optimize import curve_fit



if __name__=="__main__":

    allstart = timer()

    # load data
    path = './'
    print(path)
    #%% read dump file
    reset = False

    '''reading infiles'''

    start = timer()
    infile = glob.glob(path+'*.in')
    assert len(infile) == 1, "need to have one specified input file"

    lammps_params = read_infile(infile[0])
    thermo_header, thermo = read_thermo(path+'log.lammps')
    time_str = get_thermo_time(path+'log.lammps')

    filename = path+'out.dump'
    if os.path.exists(path+'datapts.npy'):
        multiple = np.load(path+'datapts.npy')
        ts = np.load(path+'times.npy')
    else:
        multiple, ts = read_dump(filename)
        np.save(path+'datapts.npy',multiple)
        np.save(path+'times.npy',ts)


    config = json.load(open('config.json','r'))
    params = config['params']

    a_hc = params['particle_radius']
    kT = params['temperature']*kb
    a_eff = getAEff(params)

    pnum = multiple.shape[1]
    # dt = lammps_params['timestep']*tau # [s]

    #kappa = lammps_params['kappa_2a']/(2*a_hc)
    #bpp = lammps_params['bpp']
    dt = lammps_params['timestep']

    times = ts*dt

    if lammps_params['rad'] is None:
        shell_radii = np.linalg.norm(multiple,axis=-1).mean(axis=-1)
        shell_radius = shell_radii.min() #get stand in value for rough approximation
        all_eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radii**2)
        eta_eff = all_eta_eff.max() #get stand in value for rough approximation
        title = (f"N: {pnum}, "
                 +f"R: {shell_radii.max():0.3f}-{shell_radii.min():0.3f} "
                 +r"[2a], $\eta_{eff}$:" 
                 + f"{all_eta_eff.max():0.3f}-{all_eta_eff.min():0.3f}")
    else:
        shell_radius = lammps_params['rad']
        eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radius**2)
        title = (f"N: {pnum}, "
                 +f"R: {shell_radius:0.3f} "
                 +r"[2a], $\eta_{eff}$:" 
                 + f"{eta_eff:0.3f}")

    end = timer()
    interval = end - start
    print(f"read and process files {interval:.2f}s")

    #%% 
    start = timer()
    taus = thermo[:,0]-thermo[:,0].min()
    all_taus = np.linspace(0, thermo[:,0].max(), num=150)

    samples = 10001
    for bw in [0.001]:

        #getting random sample frames
        fnum = multiple.shape[0]
        rng = np.random.default_rng()
        idx = np.arange(fnum)
        rng.shuffle(idx)
        curr_idx = idx[:samples]
        reduced = multiple[sorted(curr_idx)]

        #get g(r)
        vals,mids,bins = g_r(reduced,shell_radius=shell_radius,bin_width=bw)
        
        #save images and numpy arrays
        np.save(path+f'RDF_bw{bw}.npy',np.array([mids,vals]))
        fig,ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(r"Arclength [$\sigma$]")
        ax.set_ylabel(r"$g(r)$")
        ax.plot(mids,vals,label=f"Bin Width = {bw}")
        ax.legend()
        fig.savefig(path+f"g(r)_bw{bw}.jpg",bbox_inches='tight')
        ax.set_xlim([0,5])
        fig.savefig(path+f"g(r)_close_bw{bw}.jpg",bbox_inches='tight')
        # code to integrate the first peak and also get the peak height by multiple methods
        #handle.dumpDictionaryJSON(output,"RDF")
    
    allend = timer()
    print(f"full runtime {allend-allstart:.2f}s")
