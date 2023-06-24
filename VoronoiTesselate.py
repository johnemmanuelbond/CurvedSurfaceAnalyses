# -*- coding: utf-8 -*-
"""
Created on Fri, Jun 23, 2023

Determines whether a system is flat or spherical and performs the appropriate voronoi tesselation

@author: Jack Bond
"""

import os
import numpy as np
from main_lib.OrderParameters import vor_coord

if __name__ == "__main__"
    
    assert os.path.exists('datapts.npy'), "please run some postprocessing code to produce numpy files"
    coords = np.load('datapts.npy')
    fnum,pnum,_ = coords.shape

    #determine if system is flat
    flat = np.std(np.linalg.norm(coords[0],axis=-1)) > 0.1

    #run appropriate voronoi tesselation
    if os.path.exists('datapts_pbc.npy'):
        coords_raw = np.load('coords_raw.npy')
        box_basis = np.load('box.npy')
        vc = np.array([vor_coord(frame, flat=flat, box_basis=box_basis, tol=1e-5) for frame in coords_raw])
    else:
        vc = np.array([vor_coord(frame, flat=flat, tol=1e-5) for frame in coords])

    np.save("vor_coord.npy",np.array(vc))

    #collect topological charge metrics
    frac5 = np.mean(vc==5)
    frac7 = np.mean(vc==7)
    fracd = np.mean(vc!=6)
    q_xs = ((np.abs(vc-6).sum()/fnum)-12)/24

    output = {
            'frac5': frac5,
            'frac7': frac7,
            'fracd': fracd,
            'q_xs': q_xs,
    }

    dump_json(output, 'defects.json')