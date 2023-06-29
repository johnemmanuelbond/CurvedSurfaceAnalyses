# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 3, 2022

Used to categorize a batch of MC results, extract relevant infromation for
the reader, compute density profiles, compare them to theory, make movies,
and examine potential order parameters

@author: Jack Bond
"""

import numpy as np

from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform


def g_r(coords, bin_width=0.01, subset=None):
    """
    Given a set of frames, computed the average radial distribution function
    biw_width: determine the width in [2a] units of each rdf bin
    subset allows g_r to compute the pair correlation distribution between 
    particles in the subset and the entire ensemble. This provides a gauge on
    local microstructure
    author: Jack Bond
    """
    fnum,pnum,_ = coords.shape

    #detect if flat
    shell_rad=None
    flat = np.std(np.linalg.norm(coords[:20],axis=-1)) > 0.1
    if not flat: shell_rad = np.mean(np.linalg.norm(coords[:20],axis=-1))

    #set up subset using setdiff1d, the inverse set is easier to calculate on
    if subset is None:
        subset = np.arange(pnum)
    anti = np.setdiff1d(np.arange(pnum),subset)

    #run appropriate function
    if flat:
        return _g_r_sphere(coords,bin_width=bin_width,exclude=anti)
    else:
        return _g_r_curved(coords,shell_radius=shell_rad,bin_width=bin_width,exclude=anti)


def _g_r_sphere(coords, shell_radius=None, bin_width=0.01, exclude=None):
    """
    Given a set of frames, computed the average radial distribution function
    on a curved surface using the appropriate geodesic bins
    exclude is a helper kwarg for the subset functionality in g_r
    source: general_analysis, 7/23/22
    author: Alex yeh, Jack Bond
    """
    if shell_radius is None:
        # get mean radius over trajectory
        shell_radius = np.linalg.norm(coords[:20], axis=-1).mean()
    
    #create the right size matrix to hold the interparticle distances between particles not in exclude and all particles
    fnum, pnum, _ = coords.shape
    rshape = np.ones((pnum,pnum))
    rshape[:,exclude]=0
    rnum = int(np.sum(np.triu(rshape,k=1)))
    
    allrs = np.zeros((fnum, rnum))
    for t, frame in enumerate(coords):
        #compute geodesic distances
        cos_dists = 1-pdist(frame,metric='cosine')
        cos_dists[cos_dists>1] = 1
        cos_dists[cos_dists<-1]=-1
        #reorganize into 2d matrix
        dists = squareform(shell_radius*np.arccos(cos_dists))
        #neglect particles within exlcude along one axis
        dists[:,exclude] = 0
        #neglect double-counts
        dists = np.triu(dists)
        allrs[t,:] = dists[dists>0]
    
    #bin distances into a probability distribution
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


def _g_r_flat(coords, bin_width=0.01, exclude=None):
    """
    Given a frame or set of frames, computed the average radial distribution function
    for the simple case of a 2D flat surface
    exclude is a helper kwarg for the subset functionality in g_r
    author: Jack Bond
    """

    #create the right size matrix to hold the interparticle distances between particles not in exclude and all particles
    fnum, pnum, _ = coords.shape
    rshape = np.ones((pnum,pnum))
    rshape[:,exclude]=0
    rnum = int(np.sum(np.triu(rshape,k=1)))
    
    extent = max(pdist(coords[0]))
    
    allrs = np.zeros((fnum, rnum))
    for t, frame in enumerate(coords):
        #compute interparticledistance matrix
        dists = squareform(pdist(frame))
        #neglect particles within exlcude along one axis
        dists[:,exclude] = 0
        #neglect double-counts
        dists = np.triu(dists)
        allrs[t,:] = dists[dists>0]
        
    bins = np.histogram_bin_edges(allrs[0], bins = int(extent/bin_width), range = (0, extent))
    width = bins[1] - bins[0]
    mids = bins[:-1] + width/2
    hval = np.zeros_like(mids)
    
    counts, _ = np.histogram(allrs, bins=bins)
    vals = counts/(fnum*allrs.shape[1])

    return vals, mids, bins


def firstCoordinationShell(frames):
    """
    From a frame or trajecectory of frames, find the minimum in the radial distribution
    for the purposes of finding neighbors. Depending on the topology of the surface, this
    function outputs a distance in either a euclidean or geodesic metric. Users should be
    aware of the context.
    author: Jack Bond
    """
    if len(frames.shape) < 3:
        frames = np.array([frames])
    vals, mids, _ = g_r(frames)
    peaks, _ = find_peaks(vals)
    spacing = mids[peaks[0]]

    relevant = (mids>spacing)*(mids<2*spacing)
    relevantMids = mids[relevant]
    relevantHval = vals[relevant]
    shell_radius = relevantMids[np.argmin(relevantHval)]

    return shell_radius


def exchange_finder(frames, tol=0.05):
    """
    From a set of particle coordinates, locate where particles exchange between coordination
    shells by creating a subset of coordinates whose displacements are near the minimum in
    the radial distribution function.
    author: Jack Bond
    """

    if len(frames.shape)<3:
        frames = np.array([frames])
    
    """depending on the topology rdf min can either be in a euclidean
    or geodesic metric. But this effect is small at small distances,
    and the specific value of rdf min doesn't matter too much to the
    functionality of this method"""
    rdf_min = firstCoordinationShell(frames)
    print(rdf_min)
    coords = []

    if flat:
        for t, frame in enumerate(frames):
            #assemble matrix of distances
            dists = squareform(pdist(frame))
            #find the ones near the rdf min
            exchanges = np.where(np.abs(dists-rdf_min)<tol)
            #find the particles cooresponding to those exchanges
            ptcls = np.unique(exchanges)
            #add their coordinates to the array
            coords.append(frame[ptcls])
    else:
        shell_radius = np.linalg.norm(frames, axis=-1).mean()
        for t, frame in enumerate(frames):
            cos_dists = 1-pdist(frame,metric='cosine')
            cos_dists[cos_dists>1] = 1
            cos_dists[cos_dists<-1]=-1
            dists = shell_radius*squareform(cos_dists)
            exchanges = np.where(np.abs(dists-rdf_min)<tol)
            ptcls = np.unique(exchanges)
            coords.append(frame[ptcls])

    #hopefully coords has all the particles who exchange shells
    return coords


#########################
#WIP: CODE FOR SMOOTHING AND FINDING THE PEAK OF A g(r)
#########################


def pair_charge_correlation(frame, coord_num, q1=1, q2=1, bin_width=2):
    """
    Given a single frame, spatially correlate a set of topological charges (or, more simply, coordination number)
    Currently only functional on spherical topologies.
    source: MCBatchAnalyzer, 12/19/23
    author: Jack Bond
    see Guerra, Chaikin et. al. Nature http://www.nature.com/doifinder/10.1038/nature25468
    """

    assert np.std(np.linalg.norm(frame,axis=-1)) < 0.1, 'this method only works on spherical frames'
    shell_radius = np.linalg.norm(frame, axis=-1)

    #define histogram bins
    hbin_edge = np.histogram_bin_edges(range(10),
                               bins=int(np.pi*shell_radius/bin_width),
                               range=(0, np.pi*shell_radius))

    widths = hbin_edge[1:] - hbin_edge[:-1]
    mids = hbin_edge[:-1] + widths/2
    hval = 0*mids

    #define the topological charge from the coordination number
    qs = 6-coord_num

    #compute proper normalziation factor
    q1s = frame[qs==q1]
    Nq1 = len(q1s)
    q2s = frame[qs==q2]
    Nq2 = len(q2s)
    norm = Nq1*Nq2-Nq1*(q1==q2)

    #find subtended angles between particles with charge q1 and with charge q2
    thetas = np.einsum("ik,i,jk,j->ij",q1s,1/np.linalg.norm(q1s,axis=-1),q2s,1/np.linalg.norm(q2s,axis=-1))
    thetas[thetas>1.0]=1.0
    thetas[thetas<-1.0]=-1.0

    #convert angles to arclengths and bin using histogram
    ws = shell_radius*np.arccos(thetas).flatten()        
    counts,_ = np.histogram(ws, bins=hbin_edge)
    hval = counts/(norm/2*(np.cos((mids-widths/2)/shell_radius)-np.cos((mids+widths/2)/shell_radius)))

    return mids, hval, hbin_edge, qs


def scar_correlation(frame,charge_to_correlate = 1, bin_width=2,tol=1e-6):
    """
    Given a frame, spatially correlate the centers of mass of scars, i.e. linked sets of topological charges
    with net charge of +1
    can also correlate linked sets of topological defects with other net charges using charge_to_correlate
    Currently only functional on spherical topologies
    source: MCBatchAnalyzer, 12/19/23
    author: Jack Bond
    """

    assert np.std(np.linalg.norm(frame,axis=-1)) < 0.1, 'this method only works on spherical frames'
    shell_radius = np.linalg.norm(frame, axis=-1)

    #define histogram bins
    hbin_edge = np.histogram_bin_edges(range(10),
                               bins=int(np.pi*shell_radius/bin_width),
                               range=(0, np.pi*shell_radius))

    widths = hbin_edge[1:] - hbin_edge[:-1]
    mids = hbin_edge[:-1] + widths/2
    hval = 0*mids

    coord_shell = firstCoordinationShell(frame)

    #find_scars assembles lists of topological charges within a coord_shell of each other
    from OrderParameters import find_scars
    scars, scar_charges =find_scars(frame,tol=tol,coord_shell=coord_shell)

    print(scars, np.where(scar_charges==charge_to_correlate)[0])
    #pick out scars with the desired net charge
    charged_scars = [scars[i] for i in np.where(scarCharges==charge_to_correlate)[0]]

    #find the center of mass (average position) of each scar
    scar_coms_raw = np.array([np.mean(frame[scar],axis=0) for scar in charged_scars])
    #put charge center of mass on the sphere surface
    scar_coms = np.array([shell_radius*p/np.linalg.norm(p,axis=-1) for p in scar_coms_raw])
    
    Nscar = len(scar_coms)

    norm = Nscar*(Nscar-1)

    #compute angles between scar centers of mass
    thetas = np.einsum("ik,i,jk,j->ij",scar_coms,1/np.linalg.norm(scar_coms,axis=-1),scar_coms,1/np.linalg.norm(scar_coms,axis=-1))
    thetas[thetas>1.0]=1.0
    thetas[thetas<-1.0]=-1.0

    #compute to arclengths and bin into a histogram
    ws = shell_radius*np.arccos(thetas).flatten()        
    counts,_ = np.histogram(ws, bins=hbin_edge)
    hval = counts/(norm/2*(np.cos((mids-widths/2)/shell_radius)-np.cos((mids+widths/2)/shell_radius)))

    return mids, hval, scars, scar_coms


if __name__=="__main__":

    from timeit import default_timer as timer

    start = timer()


    end = timer()
    print(f"{end-start}s runtime")


#DEPRECATED, MAYBE WORTH RESURRECTING AT SOME POINT
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
