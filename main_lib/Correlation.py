# -*- coding: utf-8 -*-
"""
Created on Thu, Mar 3, 2022

Performs various spatial correlations on results from lammps or hoomd simulations
(or custom MC code) once they've been processed into numpy arrays

@author: Jack Bond
"""

import numpy as np

from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform

from GeometryHelpers import expand_around_pbc, hoomd_matrix_to_box

def g_r(coords, bin_width=0.01, subset=None, box=None):
    """
    Given a set of frames, computes the average radial distribution function
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
        from freud.density import RDF
        #need to consider nearest images to get the correct stats on interparticle distances
        assert not (box is None), 'please supply simulation box'
        area = np.linalg.norm(np.cross(box[0],box[1]))
        max_r = min(10,np.sqrt(area/8))

        #freud g(r) doesn't have the same subset functionality unfortunately
        freud_rdf = RDF(int(max_r/bin_width),max_r)
        hbox = hoomd_matrix_to_box(box)
        [freud_rdf.compute(system=(hbox,pts),reset=False) for pts in coords]
        bins = np.float64(np.array(freud_rdf.bin_edges))
        mids = np.float64(np.array(freud_rdf.bin_centers))
        vals = np.float64(np.array(freud_rdf.rdf))
        return vals, mids, bins
        
        #homebrew g(r) is slow and not working quite correctly
        #expand = np.array([expand_around_pbc(f,box) for f in coords])
        #anti = np.array([*anti,*np.arange(pnum,expand.shape[1])])
        #return _g_r_flat(expand,area,bin_width=bin_width,exclude=anti)
    else:
        return _g_r_sphere(coords,shell_radius=shell_rad,bin_width=bin_width,exclude=anti)


def _g_r_sphere(coords, shell_radius=None, bin_width=0.01, exclude=None):
    """
    Given a set of frames, computes the average radial distribution function
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
    rshape[exclude,:]=0
    rnum = int(np.sum(np.triu(rshape,k=1)))
    
    allrs = np.zeros((fnum, rnum))
    for t, frame in enumerate(coords):
        #compute geodesic distances
        cos_dists = 1-pdist(frame,metric='cosine')
        cos_dists[cos_dists>  1]=  1
        cos_dists[cos_dists< -1]= -1
        #reorganize into 2d matrix
        dists = squareform(shell_radius*np.arccos(cos_dists))
        #neglect particles within exlcude along one axis
        dists[exclude,:] = 0
        #neglect double-counts
        dists = np.triu(dists,k=1)
        allrs[t,:] = dists[dists>0]
    
    #bin distances into a probability distribution
    bins = np.histogram_bin_edges(allrs[0],
                                  bins = int(np.pi*shell_radius/bin_width),
                                  range = (0, np.pi*shell_radius))
    angle_bins = bins/shell_radius
    width = bins[1] - bins[0]
    mids = bins[:-1] + width/2
    
    counts, _ = np.histogram(allrs, bins=bins)
    norm = fnum * rnum * (np.cos(angle_bins[:-1]) - np.cos(angle_bins[1:]))/2
    vals = counts/norm
    return vals, mids, bins


def _g_r_flat(coords, plane_area, bin_width=0.01, exclude=None):
    """
    Given a frame or set of frames, computes the average radial distribution function
    for the simple case of a 2D flat surface. flat case g(r)'s need the number density
    rho to properly normalize the histogram
    exclude is a helper kwarg for the subset functionality in g_r
    author: Jack Bond
    """

    #create the right size matrix to hold the interparticle distances between particles not in exclude and all particles
    fnum, pnum_pbc, _ = coords.shape
    pnum=int(pnum_pbc/9)
    rshape = np.ones((pnum_pbc,pnum_pbc))
    rshape[exclude,:]=0
    rnum = int(np.sum(np.triu(rshape,k=1)))
    
    extent = np.sqrt(plane_area/3)
    
    allrs = np.zeros((fnum, rnum))
    for t, frame in enumerate(coords):
        #compute interparticledistance matrix
        dists = squareform(pdist(frame))
        #neglect particles within exlcude along one axis
        dists[exclude,:] = 0
        #neglect double-counts
        dists = np.triu(dists,k=1)
        allrs[t,:] = dists[dists>0]

    bins = np.histogram_bin_edges(allrs[0], bins = int(extent/bin_width), range = (0, extent))
    width = bins[1] - bins[0]
    mids = bins[:-1] + width/2
    
    counts, _ = np.histogram(allrs, bins=bins)
    norm = fnum * rnum * 1/extent**2 * (bins[1:]**2-bins[:-1]**2)
    vals = counts/norm
    return vals, mids, bins


def g_r_low(frames, areas, bin_width=0.01, low_frac = 1/2, box=None):
    """
    Given a set of frames and the voronoi areas per particles. computes the 
    average radial distribution function for only the lowest-density particles
    biw_width: determine the width in [2a] units of each rdf bin
    low_frac: determines the number of lowest-density particles per frame
    to calculate the rdf for.
    author: Jack Bond
    """

    # sorts particles by areas
    frames_copy = np.array([frame[np.argsort(a)] for frame,a in zip(frames,areas)])
    pnum = frames.shape[1]
    subset = np.arange(int((1-low_frac)*pnum), pnum)

    return g_r(frames_copy, bin_width=bin_width, subset=subset, box=box)


def first_coord_shell(frames, box=None):
    """
    From a frame or trajecectory of frames, find the minimum in the radial distribution
    for the purposes of finding neighbors. Depending on the topology of the surface, this
    function outputs a distance in either a euclidean or geodesic metric. Users should be
    aware of the context.
    author: Jack Bond
    """
    if len(frames.shape) < 3:
        frames = np.array([frames])
    vals, mids, _ = g_r(frames,box=box)
    peaks, _ = find_peaks(vals)
    spacing = mids[peaks[0]]

    relevant = (mids>spacing)*(mids<2*spacing)
    relevantMids = mids[relevant]
    relevantHval = vals[relevant]
    shell_radius = relevantMids[np.argmin(relevantHval)]

    return shell_radius


def exchange_finder(frames, tol=0.05, box=None):
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
    rdf_min = first_coord_shell(frames,box=box)
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

    coord_shell = first_coord_shell(frame)

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

    from FileHandling import read_xyz_frame, output_vis
    from UnitConversions import get_a_eff
    from timeit import default_timer as timer
    import json, glob
    import matplotlib.pyplot as plt

    start = timer()

    #load data from a preprocessed simulation
    config = json.load(open('config.json','r'))
    if 'simarg' in config:
        eta = config['simarg']['eta']
    else:
        aeff = get_a_eff(config['params'])/(2*config['params']['particle_radius'])
        eta = config['arg']['npart']*aeff**2 / (4*config['arg']['xxxradiusxxx']**2)
        rho_avg = config['arg']['npart'] / (4*np.pi*config['arg']['xxxradiusxxx']**2)

    coords = np.load('example_datapts.npy')
    fnum, pnum, _ = coords.shape
    vor = np.load('example_vor_coord.npy')
    times = np.load('times.npy')
    flat = np.linalg.norm(coords[:10],axis=-1).std() > 1.0
    if flat:
        box = np.load('box.npy')
    else:
        box=None
        shell_radius = np.linalg.norm(coords[:10],axis=-1).mean()


    #compare to ensemble chi_T from g(r)
    n_frames = min(int(fnum/2),int(5e8/pnum**2))

    from MSD import mto_msd_part
    msd_part, lagtime = mto_msd_part(coords[int(fnum/10):],times[:-int(fnum/10)],max_lag=5,orig_num=50)

    #getting random sample frames
    rng = np.random.default_rng()
    idx = np.arange(int(fnum/2-1),fnum)
    rng.shuffle(idx)
    curr_idx = idx[:n_frames]
    sample = coords[sorted(curr_idx)]
    sample_vor = vor[sorted(curr_idx)]

    xs = np.linspace(0,2,200)
    ys = np.linspace(0,1,100)

    dx = (xs[1:]-xs[:-1]).mean()
    dy = (ys[1:]-ys[:-1]).mean()

    rhos = np.zeros((len(xs[:-1]),len(ys[:-1])))
    qbars = np.zeros((len(xs[:-1]),len(ys[:-1])))
    dNs = np.zeros((len(xs[:-1]),len(ys[:-1])))
    Dests = np.zeros((len(xs[:-1]),len(ys[:-1])))

    sector_area = 5*np.pi#2*np.pi
    cutoff = np.sqrt(sector_area/np.pi)


    for i, phi in enumerate(np.pi*(xs[:-1]+dx/2)):
        for j, theta in enumerate(np.pi*(ys[:-1]+dy/2)):

            point = shell_radius*np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
            
            inside = np.array([ (np.linalg.norm(frame-point,axis=-1) < cutoff) for frame in sample])

            Ns = inside.sum(axis=-1)

            Qs = np.array([np.abs(sample_vor[k,idx]-6).sum() for k, idx in enumerate(inside)])

            rhos[i,j] = (Ns/sector_area).mean()
            dNs[i,j] = Ns.std()
            qbars[i,j] = (Qs/Ns).mean()

            all_ptls = np.unique(np.where(inside)[1])
            traj = msd_part[:,all_ptls].mean(axis=-1)
            idx = lagtime>1
            Dests[i,j] = 1/4 * (traj[idx][-1]-traj[idx][0])/(lagtime[idx][-1]-lagtime[idx][0])

    from matplotlib.colors import Normalize as c_norm

    fig, ax = plt.subplots(figsize=(6.5,3.25),dpi=600)
    #ax.set_ylabel('$\\theta$')
    #ax.set_xlabel('$\\phi$')
    ax.set_xticks([0,0.5,1,1.5,2])
    ax.set_xticklabels(['$0\\pi$','','$\\pi$','','$2\\pi$'])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['$0\\pi$','','$\\pi$'])
    ax.set_title(f"$R={shell_radius:.2f}, \\eta_{{eff}}={eta:.2f}, N={pnum}$")
    ax.invert_yaxis()
    ax.set_aspect('equal')
    grid = ax.pcolormesh(*np.meshgrid(xs,ys),qbars.T,
        cmap='Reds',norm=c_norm(0,0.5,clip=False))
    cbar = fig.colorbar(grid,ax=ax)
    cbar.set_label("$Q/N$")
    fig.savefig('TEST_qbar_map.jpg',bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(6.5,3.25),dpi=600)
    #ax.set_ylabel('$\\theta$')
    #ax.set_xlabel('$\\phi$')
    ax.set_xticks([0,0.5,1,1.5,2])
    ax.set_xticklabels(['$0\\pi$','','$\\pi$','','$2\\pi$'])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['$0\\pi$','','$\\pi$'])
    ax.set_title(f"$R={shell_radius:.2f}, \\eta_{{eff}}={eta:.2f}, N={pnum}$")
    ax.invert_yaxis()
    ax.set_aspect('equal')
    grid = ax.pcolormesh(*np.meshgrid(xs,ys),np.pi*aeff**2 * rhos.T,
        cmap='coolwarm_r',norm=c_norm(eta-0.02,eta+0.02,clip=True))
    cbar = fig.colorbar(grid,ax=ax)
    cbar.set_label("$\\eta_{{eff}}$")
    fig.savefig('TEST_rho_map.jpg',bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(6.5,3.25),dpi=600)
    #ax.set_ylabel('$\\theta$')
    #ax.set_xlabel('$\\phi$')
    ax.set_xticks([0,0.5,1,1.5,2])
    ax.set_xticklabels(['$0\\pi$','','$\\pi$','','$2\\pi$'])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['$0\\pi$','','$\\pi$'])
    ax.set_title(f"$R={shell_radius:.2f}, \\eta_{{eff}}={eta:.2f}, N={pnum}$")
    ax.invert_yaxis()
    ax.set_aspect('equal')
    grid = ax.pcolormesh(*np.meshgrid(xs,ys),dNs.T,
        cmap='Purples',norm=c_norm(0,2.0,clip=False))
    cbar = fig.colorbar(grid,ax=ax)
    cbar.set_label("$\\sigma_N$")
    fig.savefig('TEST_dN_map.jpg',bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(6.5,3.25),dpi=600)
    #ax.set_ylabel('$\\theta$')
    #ax.set_xlabel('$\\phi$')
    ax.set_xticks([0,0.5,1,1.5,2])
    ax.set_xticklabels(['$0\\pi$','','$\\pi$','','$2\\pi$'])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['$0\\pi$','','$\\pi$'])
    ax.set_title(f"$R={shell_radius:.2f}, \\eta_{{eff}}={eta:.2f}, N={pnum}$")
    ax.invert_yaxis()
    ax.set_aspect('equal')
    grid = ax.pcolormesh(*np.meshgrid(xs,ys),Dests.T,
        cmap='Blues',norm=c_norm(0,0.1,clip=False))
    cbar = fig.colorbar(grid,ax=ax)
    cbar.set_label("$D_L$ estimate")
    fig.savefig('TEST_D_map.jpg',bbox_inches='tight')

    end = timer()
    print(f"{end-start:.5f}s runtime")


# if __name__=="__main__":

#     from FileHandling import read_xyz_frame, output_vis
#     from UnitConversions import get_a_eff
#     from timeit import default_timer as timer
#     import json, glob
#     import matplotlib.pyplot as plt

#     def isothermal_compressibility(coords, area = None, point = None,rho_avg=None):

#         fnum, pnum, _ = coords.shape

#         #choose a random point from the starting frame
#         if point is None:
#             point = coords[0,np.random.choice(pnum)]
#             print(point)

#         #detect if the system is flat, and calculate the spherical shell radius if not flat.
#         flat = np.linalg.norm(coords[:10],axis=-1).std() > 1.0
#         if not flat:
#             shell_radius = np.linalg.norm(coords[:10],axis=-1).mean()
#             rho_avg = pnum/(4*np.pi*shell_radius**2)

#         #if no subset area is given, assume it's 1/8 of the sphere (or throw an error if the system isn't spherical)
#         if area is None:
#             assert not flat, "please suppy an area of subset to analyze"
#             area = np.pi*shell_radius**2

#         #this area-based cutoff holds on both spherical and flat surfaces.
#         cutoff = np.sqrt(area/np.pi)

#         Ns = np.array([ (np.linalg.norm(frame-point,axis=-1) < cutoff).sum() for frame in coords])

#         delNs = Ns - rho_avg*area
#         fluct = (delNs**2).mean()

#         print(f"{Ns.mean():.2f}, {rho_avg*area:.2f}")

#         return fluct/(rho_avg*rho_avg*area)

#     start = timer()

#     #load data from a preprocessed simulation
#     config = json.load(open('config.json','r'))
#     if 'simarg' in config:
#         eta = config['simarg']['eta']
#     else:
#         aeff = get_a_eff(config['params'])/(2*config['params']['particle_radius'])
#         eta = config['arg']['npart']*aeff**2 / (4*config['arg']['xxxradiusxxx']**2)
#         rho_avg = config['arg']['npart'] / (4*np.pi*config['arg']['xxxradiusxxx']**2)

#     coords = np.load('example_datapts.npy')
#     fnum, pnum, _ = coords.shape
#     flat = np.linalg.norm(coords[:10],axis=-1).std() > 1.0
#     if flat:
#         box = np.load('box.npy')
#     else:
#         box=None
#         shell_radius = np.linalg.norm(coords[:10],axis=-1).mean()

#     print("local chi_T",isothermal_compressibility(coords[int(fnum/3):],area=40.0))

#     #compare to ensemble chi_T from g(r)
#     n_frames = min(int(fnum/2),int(5e8/pnum**2))

#     #getting random sample frames
#     rng = np.random.default_rng()
#     idx = np.arange(int(fnum/2-1),fnum)
#     rng.shuffle(idx)
#     curr_idx = idx[:n_frames]
#     sample = coords[sorted(curr_idx)]

#     #get g(r)
#     vals,mids,bins = g_r(sample,bin_width=0.005,box=box)

#     if flat:
#         jac = 2*np.pi*mids
#     else:
#         jac = 2*np.pi*shell_radius*np.sin(mids/shell_radius)

#     integrand = vals-1

#     integral = (integrand*jac*(bins[1:]-bins[:-1])).sum()

#     print("ensemble chi_T", 1/rho_avg + integral)



#     end = timer()
#     print(f"{end-start:.5f}s runtime")
