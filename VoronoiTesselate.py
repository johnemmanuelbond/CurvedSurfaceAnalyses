# -*- coding: utf-8 -*-
"""
Created on Fri, Jun 23, 2023

Determines whether a system is flat or spherical and performs the appropriate voronoi tesselation.
Uses areas to calculate a distribution of densities.

@author: Jack Bond
"""

import os, json
import numpy as np
from main_lib.FileHandling import dump_json
from main_lib.UnitConversions import get_a_eff
from main_lib.OrderParameters import vor_coord_with_areas

if __name__ == "__main__":
    
    assert os.path.exists('datapts.npy'), "please run some postprocessing code to produce numpy files"
    coords = np.load('datapts.npy')
    fnum,pnum,_ = coords.shape
    eq_sample = np.arange(int(fnum/4),fnum)

    #determine if system is flat
    flat = np.std(np.linalg.norm(coords[0],axis=-1)) > 0.1

    #run appropriate voronoi tesselation
    if os.path.exists('datapts_pbc.npy'):
        coords_raw = np.load('datapts_pbc.npy')
        box_basis = np.load('box.npy')
        eta = json.load( open('config.json','r'))['simarg']['eta']
        #TEMP UNTIL UPDATED POSTPROCESS
        if eta == 0:
            tess = np.zeros((fnum,2,pnum))
        else:
            tess = np.array([vor_coord_with_areas(frame, flat=flat, box_basis=box_basis) for frame in coords_raw])
    else:
        #TEMP UNTIL UPDATED POSTPROCESS
        A = json.load( open('config.json','r'))['arg']['xxxpairstrengthxxx']
        if A == 0:
            tess = np.zeros((fnum,2,pnum))
        else:
            tess = np.array([vor_coord_with_areas(frame, flat=flat) for frame in coords])

    vc = tess[:,0,:]
    q = 6 - vc
    areas = tess[:,1,:]

    np.save("vor_coord.npy",np.array(vc))
    np.save("vor_areas.npy",np.array(areas))

    #assemble charge distribution
    mids = np.unique(q)
    hval = 0*mids
    for i, mid in enumerate(mids):
        hval[i] = np.mean(q[eq_sample] == mid)
    np.save('q_fracs.npy',np.array([mids,hval]))
    #NOTE: Q = np.sum(np.abs(mids)*hval)*pnum)

    #assemble density distribution
    rho_inst = 1/areas[eq_sample]
    hval,bin_edges = np.histogram(rho_inst.flatten(),bins=2000,range=(0,4/np.pi),density=True)
    widths = (bin_edges[1:]-bin_edges[:-1])/2
    mids = bin_edges[:-1]+widths
    np.save('rho_dist.npy',np.array([mids,hval]))

    #assembling rhos per charge
    mids = np.unique(q)
    hval = 0*mids
    for i, mid in enumerate(mids):
        hval[i] = np.mean(1/areas[eq_sample][q[eq_sample] == mid])
    np.save('q_rho.npy',np.array([mids,hval]))