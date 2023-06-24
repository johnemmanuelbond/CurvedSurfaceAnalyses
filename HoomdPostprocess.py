# -*- coding: utf-8 -*-
"""
Created on Fri, Jun 23, 2023

Quickly converts a hoomd gsd file into the common-ground numpy arrays we use mostly

@author: Jack Bond
"""

import os, glob, random, json
import numpy as np

import gsd.hoomd

from main_lib.GeometryHelpers import minimum_image, hoomd_box_to_matrix

if __name__=="__main__":
    
    with open('config.json','r') as f:
        config = json.load(f)
        simarg = config['simarg']
    
    with gsd.hoomd.open("./traj.gsd") as hoomd_out:
        coords_raw = np.array([f.particles.position for f in hoomd_out])
        wraps = np.array([f.particles.image for f in hoomd_out])
        times = np.array([f.configuration.step for f in hoomd_out]) * simarg['dt']
        box_basis = hoomd_box_to_matrix(hoomd_out[0].configuration.box)

    fnum,pnum,_ = coords_raw.shape
    coords = minimum_image(multiple,wraps,box_basis)

    np.save("datapts_pbc.npy",coords)
    np.save("datapts.npy",coords)
    np.save("times.npy",times)
    np.save("box.npy",box_basis)