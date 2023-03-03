# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 3, 2022

Used to categorize a batch of MC results, extract relevant infromation for
the reader, compute density profiles, compare them to theory, make movies,
and examine potential order parameters

@author: Jack Bond
"""

import numpy as np

from OrderParameters import Vc, Nc, findScars
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist

#icosohedral angles
r_ico = np.sin(2*np.pi/5)
theta1 = 2*np.arcsin(1/2/r_ico)
theta2 = 2*np.arccos((r_ico**2+r_ico**2*np.cos(theta1/2)**2-3/4)/(2*r_ico**2*np.cos(theta1/2)))

"""
Given a frame or set of frames, computed the average radial distribution function
source: general_analysis, 7/23/22
author: Alex yeh, Jack Bond
"""
def g_r(coords, shell_radius=None, bin_width=0.1, flat = False):
    """calculates the pair distribution function from the given coordinates"""
    if flat:
        return g_r_flat(coords,bin_width=bin_width)

    if shell_radius is None:
        # get mean radius over run
        shell_radius = np.linalg.norm(coords, axis=-1).mean()
        
    fnum, pnum, _ = coords.shape
    
    allrs = np.zeros((fnum, (pnum*(pnum-1)//2)))
    for t, frame in enumerate(coords):
        cos_dists = 1-pdist(frame,metric='cosine')
        cos_dists[cos_dists>1] = 1
        cos_dists[cos_dists<-1]=-1
        allrs[t,:] = shell_radius*np.arccos(cos_dists)
    
    bins = np.histogram_bin_edges(allrs[0],
                                  bins = int(np.pi*shell_radius/bin_width),
                                  range = (0, np.pi*shell_radius))
    angle_bins = bins/shell_radius
    width = bins[1] - bins[0]
    mids = bins[:-1] + width/2
    hval = np.zeros_like(mids)
    
    counts, _ = np.histogram(allrs, bins=bins)
    vals = counts/(fnum*allrs.shape[1]) * 2/(np.cos(angle_bins[:-1]) - np.cos(angle_bins[1:]))
    return vals, mids, bins

"""
Given a frame or set of frames, computed the average radial distribution function
source: Correlation, 3/3/23
author: Jack Bond
"""
def g_r_flat(coords bin_width=0.01):
        """calculates the pair distribution function from the given coordinates"""
    fnum, pnum, _ = coords.shape
    
    extent = max(pdist(coords[0]))

    allrs = np.zeros((fnum, (pnum*(pnum-1)//2)))
    for t, frame in enumerate(coords):
        dists = pdist(frame)
        allrs[t,:] = dists

    bins = np.histogram_bin_edges(allrs[0],
                                  bins = int(extent/bin_width),
                                  range = (0, extent))
    width = bins[1] - bins[0]
    mids = bins[:-1] + width/2
    hval = np.zeros_like(mids)
    
    counts, _ = np.histogram(allrs, bins=bins)
    vals = counts/(fnum*allrs.shape[1])

    return vals, mids, bins

#WIP: CODE FOR SMOOTHING AND FINDING THE PEAK OF A g(r)

"""
Given a frame, spatially correlate a set of topological charges (or, more simply, coordination number)
source: MCBatchAnalyzer, 12/19/23
author: Jack Bond
"""
def pair_charge_correlation(q1,q2,frame, shellRadius, bin_width=2): 
    qs = 6-Vc(frame,R=shellRadius)
    #qs = 6-Nc(frame,shellradius=order.firstCoordinationShell(frame)) #DEPRECATED 

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


"""
Given a frame, spatially correlate scars, i.e. linked sets of topological charges
source: MCBatchAnalyzer, 12/19/23
author: Jack Bond
"""
def scar_correlation(frame, shellRadius,charge_to_correlate = 1, bin_width=2,tol=1e-6): 
    scars, scarCharges =findScars(frame,tol=tol)

    hbin_edge = np.histogram_bin_edges(range(10),
                               bins=int(np.pi*shellRadius/bin_width),
                               range=(0, np.pi*shellRadius))

    widths = hbin_edge[1:] - hbin_edge[:-1]
    mids = hbin_edge[:-1] + widths/2
    hval = 0*mids

    print(scars, np.where(scarCharges==charge_to_correlate)[0])
    chargedScars = [scars[i] for i in np.where(scarCharges==charge_to_correlate)[0]]

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


if __name__=="__main__":

    from timeit import default_timer as timer

    start = timer()


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