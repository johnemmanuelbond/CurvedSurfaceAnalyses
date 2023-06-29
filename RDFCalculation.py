# -*- coding: utf-8 -*-
"""
Created on Fri, Jun 23, 2023

Uses the Correlation module to run a baseline radial distribution
calculation for flat and curved systems

@author: Jack Bond
"""

import os, json
import numpy as np

from main_lib.FileHandling import dump_json
from main_lib.Correlation import g_r

DEFAULT_ARGS = {
    "coord_file": "datapts.npy",
    "max_samples": 5e9,
    "bin_widths": [0.005,0.01], #in 2a units
}


if __name__=="__main__":

    if os.path.exists("RDF_analysis_arguments.json"):
        args = json.load(open('RDF_analysis_arguments.json'))
    else:
        args = DEFAULT_ARGS
        dump_json(args,"RDF_analysis_arguments.json")

    box = None
    if os.path.exists('box.npy'):
        box = np.load('box.npy')

    coords = np.load(args['coord_file'])
    fnum,pnum,_ = coords.shape

    #we want to sample interparticle distances from the last half of a simulation
    # we don't want to sample more than max_samples interparticle distances because at that point it's diminishing returns
    n_frames = min(int(fnum/2),int(args['max_samples']/pnum**2))
    output = dict()

    #do several bin widths
    for bw in args['bin_widths']:

        #getting random sample frames
        rng = np.random.default_rng()
        idx = np.arange(int(fnum/2-1),fnum)
        rng.shuffle(idx)
        curr_idx = idx[:n_frames]
        sample = coords[sorted(curr_idx)]

        #get g(r)
        vals,mids,bins = g_r(sample,bin_width=bw,box=box)
        
        #save images and numpy arrays
        np.save(f'RDF_bw{bw}.npy',np.array([mids,vals]))

        # code to integrate the first peak and also get the peak height by multiple methods

        output[f"bw={bw}"] = {
                "bw": bw,
                "gr_peak_simple": np.max(vals[mids<3]),
                "contact_simple": mids[mids<3][np.argmax(vals[mids<3])],
                "gr_peak_fit": None,
                "gr_area": None,
        }
    

    dump_json(output,"RDF.json")
