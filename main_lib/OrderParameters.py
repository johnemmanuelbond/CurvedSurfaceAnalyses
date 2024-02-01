# -*- coding: utf-8 -*-
"""
Created on Mon, Apr 25, 2022

Contains a collection of calculations which compute single value, or arrays
of order parameters on a frame of 3D coordinate data.

@author: Jack Bond
"""

import numpy as np
import scipy as sp

from scipy.spatial.distance import pdist, cosine, squareform
from scipy.spatial import SphericalVoronoi, Voronoi, ConvexHull#, geometric_slerp


from GeometryHelpers import expand_around_pbc, polygon_area
from GeometryHelpers import theta1 as theta_ico

#first coordination shell for our standard silica colloids at effective close-packing
DEFAULT_CUTOFF = (1.44635/1.4)*0.5*(1+np.sqrt(3))


def bond_order(frame, order=6,reference = np.array([0,1,0]), cutoff=DEFAULT_CUTOFF):
    """
    given a frame of 3D coordinates, calculates the bond orientational order
    parameter of each particle with respect to a reference direction
    'order': n-fold order defines the argument of the complex used
    to calculate psi_n, 
    'reference': the vector to calculate angles w.r.t. for the purpose of
    calculating psi_n
    author: Jack Bond
    """
    pnum = frame.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = frame[i]-frame[j]

    order_param = np.zeros(pnum)+np.ones(pnum)*0j
    neighbors = squareform(pdist(frame)) < cutoff

    for i in np.arange(pnum):
        psi=[]
        for j in np.where(np.linalg.norm(dr_vec[i],axis=-1)<3)[0]:
            if (i!=j) and (neighbors[i,j]):
                arg = np.arccos(1-cosine(dr_vec[i,j],reference))
                psi.append(np.exp(1j*order*arg))
        order_param[i]=np.mean(np.array(psi))

    return order_param


#create the typical array of colors used to indicate topological charge
from matplotlib.colors import to_rgb
cols = ['grey' for _ in range(13)]
cols[4] = 'orange'
cols[5] = 'red'
cols[6] = 'white'
cols[7] = 'green'
cols[8] = 'blue'
cols[0] = 'black'
cols[-1] = 'black'
VORONOI_COLORS = np.array([to_rgb(c) for c in cols])

def _calculate_planar_voronoi_areas(points, regions, vertices):
    """
    Given the outputs from a scipy.spatial.Voronoi object, returns
    the area afforded to each point.
    Modified from scipy.spatial._spherical_voronoi.py in scipy source code
    'points': (N,3) array of particle positions
    'regions': (N, *) list of lists, each list contains indices of 'vertices'
    corresponding to that particle's region
    of 'vertices' 
    'vertices': (nvertices,3) array of all the vertices for this object
    author: Jack Bond
    """
    sizes = [len(region) for region in regions]
    csizes = np.cumsum(sizes)
    num_regions = csizes[-1]

    # We create a set of triangles consisting of one point and two Voronoi
    # vertices. The vertices of each triangle are adjacent in the sorted
    # regions list.
    point_indices = [i for i, size in enumerate(sizes)
                     for j in range(size)]

    nbrs1 = np.array([r for region in regions for r in region])

    # The calculation of nbrs2 is a vectorized version of:
    # np.array([r for region in regions for r in np.roll(region, 1)])
    nbrs2 = np.roll(nbrs1, 1)
    indices = np.roll(csizes, 1)
    indices[0] = 0
    nbrs2[indices] = nbrs1[csizes - 1]

    # Create the complete set of triangles and calculate their areas
    triangles = np.hstack([points[point_indices],
                           vertices[nbrs1],
                           vertices[nbrs2]
                           ]).reshape((num_regions, 3, 3))
    triangles[:,:,2]=1
    tri_areas = np.abs(np.linalg.det(triangles)/2)

    # Sum the areas of the triangles in each region
    areas = np.cumsum(tri_areas)[csizes - 1]
    areas[1:] -= areas[:-1]

    return areas

def vor_tesselate(frame, flat=None, R=None, tol=1e-5, box_basis=None, pbc_padfrac=0.8, return_areas=False):
    """
    given a frame of x,y,z coordinates, performs a voronoi tesselation.
    Returns the coordinates of the voronoi vertices, as well as a boolean array
    that maps each particle to the indices of its vertices.
    The vertex map contains the coordination number of each particle, and can
    also be used to determine nearest neighbors (use a cosine metric to determine
    shared vertices)
     the coordination number of each particle using a voronoi tesselation.
    'flat': determine whether the frame in question sits on a spherical
    surface or not, will detect for itself if unspecified.
    'R': specify a radius for an underlying spherical surface. Will calculate
    if unspecified
    'tol': The tolerance passed to SphericalVoronoi. Basically it's how far
    from the spherical surface each particle is allowed to be at maximum.
    'box_basis': specify the basis vectors for period boundary conditions
    'pbc_padfrac': specifies the proportion of extra particles needed to respect
    periodic boundary conditions.
    'return_areas': if True, will also return the areas associated with each
    particle's voronoi polygon.
    author: Jack Bond
    """

    pnum, _ = frame.shape
    if flat is None:
        flat = (np.std(np.linalg.norm(frame,axis=-1)) > 0.1)
    
    if flat:
        if box_basis is None: raise Exception("please input simulation box")
        sv = Voronoi(expand_around_pbc(frame,box_basis,padfrac=pbc_padfrac)[:,:2])
        vtx = sv.vertces
        reg = [sv.regions[i] for i in sv.point_region[:pnum]]

        if return_areas:
            if -1 in [r for rs in reg for r in rs]:
                print('tesselation includes points off the grid')
                areas = np.array([polygon_area(vertices[r]) for r in reg])
            else:
                areas = _calculate_planar_voronoi_areas(frame,ref,vtx)
    
    else:
        minZ = min(frame[:,2])
        if R == None:
            radius = np.mean(np.linalg.norm(frame,axis=1))
        else:
            radius = R
        sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
        sv.sort_vertices_of_regions()
        vtx = sv.vertices
        reg  = sv.regions

        if return_areas:
            areas = sv.calculate_areas()

    vtx_map = np.full((pnum,len(vtxs)),False)
    for i, r in enumerate(reg):
        vtx_map[i][r]=True

    if return_areas:
        return vtx, vtx_map, areas
    else: return vtx, vtx_map

def vor_coord(frame,box_basis=None):
    """
    given a frame of x,y,z coordinates, returns the coordinationn number
    of each particle using the voronoi tesselation.
    'box_basis': specify the basis vectors for period boundary conditions
    For more tunable kwargs simply use vor_tesselate 
    author: Jack Bond
    """
    vtx_map = vor_tesselate(frame,box_basis=box_basis)[1]
    return np.sum(vtx_map,axis=-1)


def vor_neighbors(frame, box_basis=None):
    """
    given a frame of x,y,z coordinates, returns a NxN boolean array of
    which particles share a voronoi vertex
    'box_basis': specify the basis vectors for period boundary conditions
    For more tunable kwargs simply use vor_tesselate 
    author: Jack Bond
    """
    vtx_map = voronoi_tesselate(frame, box_basis=box_basis)[1]
    return squareform(pdist(vtx_map,'cosine'))<1 #if the dot product is zero they don't share a vertex


def coordination_shell_densities(frame, areas=None, neighbors = None, box_basis=None, cutoff=DEFAULT_CUTOFF):
    """
    Given a frame, calclates the point-density based on the area of voronoi
    polygons INCLUDING NEAREST NEIGHBORS. Determines nearest neighbors via a 
    simple cutoff radius. Returns the local density per particle, the total
    area of of each particle's coordination shell, and the number of particles
    therein.
    'areas': an option to provide a pre-calculated voronoi areas with
    custom voronoi parameters
    'neighbors': An NxN boolean array indicating which particles are neighbors
    'box_basis': Needed to calculate voronoi areas if they're not provided.
    'cutoff': Needed to calculate neighbor array if you want to avoid running the
    voronoi calculation.
    author: Jack Bond
    """

    pnum, _ = frame.shape
    if areas is None:
        _, vtx_map, areas = vor_tesselate(frame,box_basis=box_basis,return_areas=True)
        neighbors = squareform(pdist(vtx_map,'cosine'))<1
    elif neighbors is None:
        neighbors = neighbors(frame,cutoff=cutoff)

    for i, nei in enumerate(neighbors[:pnum]):
        #Nc does not include the particle itself when counting neighbors
        As = areas[[i,*(np.where(nei!=0)[0]%pnum)]]
        shell_areas[i] = As.sum()
        shell_counts[i] = As.size

    return shell_counts/shell_areas, shell_areas, shell_counts


def plot_voronoi_sphere(points, tol=1e-5, colors = VORONOI_COLORS, filename="voronoi_frame.jpg", view = np.array([0,0,1])):
    """
    Given a spherical frame, plots the centers and vertices of the voronoi tesselation on the surface of a sphere.
    'view'; the vector along which we view the sphere
    source: plot_voronoi, 5/26/23
    author: John Edison
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    radius = np.linalg.norm(points,axis=-1).mean()
    sv     = SphericalVoronoi(points, radius = radius,threshold=tol)
    sv.sort_vertices_of_regions()
    areas = sv.calculate_areas()

    #set up axes with approprite view
    fig = plt.figure(figsize=(7,7),dpi=600)
    ax = fig.add_subplot(projection='3d')
    front = points[points@view>0]
    theta = np.arccos( min(max(view[2]/np.linalg.norm(view),-1),1) )
    if theta == 0 or theta == np.pi:
        phi=0
    else:
        if view[0]==0:
            phi = np.pi/2 + np.pi*(view[1]<0)
        else:
            phi = np.arctan(view[1]/view[0])

    ax.view_init(90-180*theta/np.pi,180*phi/np.pi)
    plt.axis('off')
    
    #plotting each point and voronoi polygon
    for i, region in enumerate(sv.regions):

        if (areas[i] > 1.50*np.pi):
            continue
        
        n = len(region)
        vtx = sv.vertices[region]
        
        tri = Poly3DCollection([vtx])

        if (n < len(colors)):
            tri.set_color(colors[n])
        else:
            tri.set_color('k')
        tri.set_edgecolor('k')
        tri.set_linewidth(0.5)
        ax.add_collection3d(tri)

    ax.scatter(*(front.T), c='k',s=2)

    ax.autoscale()
    ax.set_xlim3d(-radius,radius)     #Reproduce magnification
    ax.set_ylim3d(-radius,radius)     #...
    ax.set_zlim3d(-radius, radius)
    ax.set_aspect('equal')

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_zticks([])

    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


def plot_voronoi_plane(points, box=None, colors = VORONOI_COLORS, filename="voronoi_frame.jpg"):
    """
    Given a planar frame, plots the centers and vertices of the voronoi tesselation
    source: plot_voronoi, 5/26/23
    author: John Edison, Jack Bond
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if box is None: raise Exception('please input simulation box')
    pnum,_= points.shape

    vor = Voronoi(expand_around_pbc(points,box)[:,:2])

    fig = plt.figure(figsize=(7,7),dpi=600)
    ax = fig.add_subplot()

    plt.axis('off')

    for i, pr in enumerate(vor.point_region[:pnum]):
        region = vor.regions[pr]
        n = len(region)
        vtx = vor.vertices[region]
        
        if -1 in region:
            continue

        tri = PolyCollection([vtx])

        if (n < len(colors)):
            tri.set_color(colors[n])
        else:
            tri.set_color('k')
        tri.set_edgecolor('k')
        tri.set_linewidth(0.5)
        ax.add_collection(tri)

    ax.scatter(*(points[:,:2].T), c='k',s=1)
    ax.autoscale()
    ax.set_aspect('equal')

    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

    fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)


def cluster_defects(frame, vtx_map = None, box_basis=None):
    """
    given a frame, finds and links together clusters of neighboring
    topological defects (non-6-coordinated particles). Determines neighbors via
    shared voronoi vertices.
    Assigns each particle a number corresponding to which cluster it belongs to.
    Uncharged particles are labeled '0'
    'vtx_map': if you've already ran vor_tesselate, you can input the vertex map here
    'box_basis': needed if you need to recalculate the vertex map
    author: Jack Bond
    """
    if vtx_map is None: vtx_map = voronoi_tesselate(frame, box_basis=box_basis)

    #determine which particles are both charged and neighboring
    neighbors = 1*(squareform(pdist(vtx_map,'cosine'))<1) #if the dot product is zero they don't share a vertex
    vor = np.sum(vtx_map,axis=-1)
    charges = np.outer((vor!=6),(vor!=6))
    mat = (charges*neighbors)

    #transforms charged neighbor array into rref form without empty slices.
    #this is okay because the order particles within each slice is unaffected.
    first_digit = np.argmax(mat,axis=1)
    rref = mat[np.argsort(first_digit)]
    links = rref[np.sum(rref,axis=-1)>0]

    #iterates over the rref-formatted 'links' combining lists together which share
    #a filled index. Does so until 'links' stops changing.
    for i in range(100):
        shared = np.array([[1*np.any(l1*l2) for l1 in links] for l2 in links])
        #print(shared)

        new_links = np.array([1*(np.sum(links[s>0],axis=0)>0) for s in shared])
        # print(new_links)
        new_links = np.unique(new_links, axis=0)
        # print(new_links)
        if new_links.shape==links.shape:
            if np.any(new_links==links):
                #print(i)
                break
        links = new_links.copy()

    links = new_links
    clust_number = np.sum([(i+1)*l for i,l in enumerate(links)],axis=0)
    return clust_number


def g_ico(frame,vor=None):
    """
    Given an Nx3 frame on a spherical surface, calculate the icoashedral order parameter.
    This parameter reflects icosahedral ordering of defects by finding a peak of defect-defect
    correlations at the first icosahedral angle compared to the inter-icoashedral angle.
    'vor': the voronoi coordination number of each particle, will calculate if not supplied.
    author: Jack Bond
    """
    if vor is None: vor = vor_coord(frame, box_basis=None)
    R = np.linalg.norm(frame,axis=-1).mean()
    defects = frame[vor!=6]
    dists = pdist(defects)
    arcs = 2*np.arcsin(dists/R/2)
    bin_edges = np.array([1,3,5])*theta_ico/4
    heights,_ = np.histogram(arcs,bins=bin_edges)
    norm = (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:]))/2
    vals = heights/norm

    return 2*(1-vals[0]/vals[1])



#CURRENTLY DEPRECATED, NEED TO REVISIT
def Psi6(frame, reference = np.array([0,1,0]),coord_shell=DEFAULT_CUTOFF):
    """
    meant to do local bond orientational order, psi6 on the sphere by choosing
    a self-consistent set of reference vectors via 3D rotations.
    author: Jack Bond
    """
    N = frame.shape[0]
    i,j = np.mgrid[0:N,0:N]
    _, n = coord(frame,cutoff=coord_shell)
    vc = vor_coord(frame)


    # finding the tanget-plane components of the nearest-neighbor bonds, we'll
    # call these unit vectors the bond orientations
    dr_vec = frame[i]-frame[j]
    bond_orientations = []
    for k, r in enumerate(frame):
        normal = r/np.linalg.norm(r)
        drk = dr_vec[k]
        bonds = drk[n[k]==1]
        bonds = bonds - np.array([np.dot(b,normal)*normal for b in bonds])
        orientation = np.array([b/np.linalg.norm(b,axis=-1) for b in bonds])
        bond_orientations.append(orientation)


    # bond orientation depends on a global reference vector. On the surface of
    # a sphere we parallel transport said reference vector along a geodesic
    # to the particle who's bonds we're trying to measure
    Rs = np.linalg.norm(frame,axis=-1)
    polar = np.array(np.arccos(frame[:,2]/Rs))
    azimuth = np.array(np.arctan(frame[:,1]/(frame[:,0]+1e-20))+np.pi*(frame[:,0]<0))

    axes = np.array([-1*np.sin(azimuth),np.cos(azimuth),0*azimuth]).T
    
    iden = np.array([np.eye(3) for ax in axes])
    outer = np.array([np.outer(ax,ax) for ax in axes])
    cross = np.array([np.array([[0*ax[0],-1*ax[2],ax[1]],
                                [ax[2],0*ax[0],-1*ax[0]],
                                [-1*ax[1],ax[0],0*ax[0]]]) for ax in axes])

    rotationMatrices = np.einsum('k,kij->kij',np.cos(polar),iden)
    rotationMatrices += np.einsum('k,kij->kij',np.sin(polar),cross)
    rotationMatrices += np.einsum('k,kij->kij',(1-np.cos(polar)),outer)


    references = np.einsum('aij,j->ai',rotationMatrices,reference)
    
    LC = np.zeros((3,3,3))
    LC[0,1,2],LC[1,2,0],LC[2,0,1] = 1.0,1.0,1.0
    LC[2,1,0],LC[0,2,1],LC[1,0,2] = -1.0,-1.0,-1.0
    #print(LC)

    references2 = np.einsum('ai,aj,j->ai',axes,axes,reference)
    references2 += np.einsum('a,ijk,ilm,al,m,ak->ai',np.cos(polar),LC,LC,axes,reference,axes)
    references2 += np.einsum('a,ijk,aj,k->ai',np.sin(polar),LC,axes,reference)

    #print(references-references2)


    #normals1 = np.einsum('aij,j->ai',rotationMatrices,np.array([0,0,1]))
    #normals2 = frame/np.array([Rs,Rs,Rs]).T
    #print(normals1-normals2)

    # with the bond orientations and the appriopriate reference vectors chosen,
    # the calculation of the 6-fold bond orientation order parameter is easy
    psi6 = np.zeros(N,dtype=complex)
    for k, bos in enumerate(bond_orientations):
        ref = references[k]
        v = vc[k]
        argument = np.array([np.arccos(np.dot(bo,ref)) for bo in bos])
        psi6[k] = np.sum(np.exp(6j*argument))/v

    return psi6, np.mean(np.abs(psi6))


#CURRENTLY BROKEN
def C6(frame,cutoff=DEFAULT_CUTOFF):
    """
    uses the local psi6 to compute crystalline connectivity C6
    author: Jack Bond
    """

    pnum,_ = frame.shape
    
    # computing the reference c6 for a perfect lattice
    # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
    shells = -1/2 + np.sqrt((pnum-1)/3 + 1/4)
    # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
    c6_hex = 6*(3*shells**2 + shells)/pnum

    n, vc = coord(frame,cutoff=cutoff)
    
    psi6, psi6global = Psi6(frame)
    C6 = 0*vc

    for k, v in enumerate(vc):
        pi = psi6[k]
        pj = np.conjugate(psi6[n[k]==1])
        chi = np.abs(np.real(pi*pj))/np.abs(pi*pj)
        C6[k] = np.sum(chi>=0.32)

    return C6, np.mean(C6)/c6_hex, psi6, psi6global


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

    coords = np.load('example_datapts.npy')

    plot_voronoi_sphere(coords[-1])

    #only get the last half

    fnum, pnum, _ = coords.shape
    vor = np.array([])
    areas = np.array([])


    for i,frame in enumerate(coords[int(coords.shape[0]/2):]):
        vc, ar = vor_coord_with_areas(frame)
        vor = np.array([*vor,vc])
        areas = np.array([*areas,ar])

#    vor = np.load('example_vor_coord.npy')
    vor = vor[int(vor.shape[0]/2):]

    bin_edges = np.histogram_bin_edges(np.random.rand(10),bins = 50, range=(0,1))
    width = bin_edges[1]-bin_edges[0]
    mids = bin_edges[:-1]+width/2

    # calculate f5 and f7 distributions
    f5 = np.mean(vor==5,axis=0)
    f7 = np.mean(vor==7,axis=0)

    fig, ax = plt.subplots(figsize=(3.25,3.25),dpi=600)
    ax.set_title(f"$\eta_{{eff}}={eta:.2f}$")

    vals, _ = np.histogram(f5,bins=bin_edges)
    prob = vals/pnum
    ax.plot(mids,prob,color='red',label='$p(f_5)$',lw=0.7)
    avg = np.sum(mids*prob)
    ax.axvline(x=avg,color='red',lw=0.5,ls='--')

    mobile = (f5 >= avg)
    immobile = (f5 < avg)
    print(f"{mobile.sum()} 'mobile' particles")

    vals, _ = np.histogram(f7,bins=bin_edges)
    prob = vals/pnum
    ax.plot(mids,prob,color='green',label='$p(f_7)$',lw=0.7)
    avg = np.sum(mids*prob)
    ax.axvline(x=avg,color='green',lw=0.5,ls='--')

    ax.legend()
    fig.savefig('time_average_charge_distributions.jpg',bbox_inches='tight')

    # threshold based on distribution moments?
    last_section = 1.0
    sample_rate=20
    idx = np.arange(int((1-last_section)*fnum),fnum,sample_rate)
    colors = np.array([np.array([immobile,immobile,immobile]).T for _ in range(fnum)])
    output_vis("TEST_threshold.atom",coords[idx],colors=colors[idx],show_shell=True)

    # calculate eta on either side using voronoi
    eta_low = (np.pi*aeff**2) * np.sum(mobile)/np.sum(areas[:,mobile],axis=-1).mean()
    eta_high = (np.pi*aeff**2) * np.sum(immobile)/np.sum(areas[:,immobile],axis=-1).mean()
    print(f"low eta: {eta_low:.4f}\nens eta: {eta:.4f}\nhigh eta:{eta_high:.4f}")

    # do MSD on each side of the threshold

    from MSD import mto_msd_part

    times = np.load('times.npy')
    msd_part, lag = mto_msd_part(coords,times,max_lag=100,delta=5)

    fig, ax = plt.subplots(figsize=(3.25,3.25),dpi=600)
    ax.set_xlabel("$t/\\tau$")
    ax.set_ylabel("$<\delta r^2>/(2a)^2$")

    ax.plot(lag, msd_part[:,mobile].mean(axis=-1),ls=':',lw=1.5,
        color='black', label = 'high defect-fraction')
    ax.plot(lag, msd_part[:,immobile].mean(axis=-1),ls=':',lw=1.5,
        color='grey', label = 'low defect-fraction')
    ax.plot(lag, msd_part.mean(axis=-1),ls=':',lw=1.5,
        color='blue',label='ensemble average')
    ax.legend()

    fig.savefig('TEST_msd_threshold.jpg',bbox_inches='tight')

    end = timer()
    print(f"{end-start:.5f}s runtime")

