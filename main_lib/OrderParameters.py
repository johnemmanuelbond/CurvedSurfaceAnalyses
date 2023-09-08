# -*- coding: utf-8 -*-
"""
Created on Mon, Apr 25, 2022

Contains a collection of calculations which compute single value, or arrays
of order parameters on a frame of 3D coordinate data.

@author: Jack Bond
"""

import numpy as np
import scipy as sp

from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi, Voronoi, ConvexHull#, geometric_slerp
from scipy.integrate import quad
from scipy.special import gamma as gammafunc

from GeometryHelpers import expand_around_pbc


def B2(phi,splits=np.array([0,5,np.infty]),dim=3, core_radius=None):
    """
    calculates the second virial coeffecient for an arbitrary potential. This
    potential, phi, returns an energy in kT units based on only a pair distance
    r, in 2a units.
    'splits': bounds on the integration, can break the integral up into
    discreet points for precision purposes.
    'dim': this method can compute B2 in arbirtrary (integer) dimensions
    'core_radius': the size of am optional hard core (phi->infinity) within
    the potential
    author: Jack Bond
    """
    
    if type(phi) != type(np.mean): raise Exception("Input must be a python function capable of acting on arrays.")

    mayer_f = lambda r: np.exp(-phi(r))-1

    #hypersphere solid angle
    g = gammafunc(dim/2+1)
    hsolid = lambda r: dim*np.pi**(dim/2)/g *r**(dim-1)

    integrand = lambda r: -1/2 * hsolid(r) * mayer_f(r)

    # infinity plays poorly, so if we want a hard core we need to start from 1.0
    if core_radius!=None:
        splits = splits[splits>=2*core_radius]
        if not (np.any(splits==2*core_radius)):
            splits = np.array([2*core_radius,*splits])

    bounds = zip(splits[:-1],splits[1:])
    parts = [quad(integrand,a,b)[0] for (a,b) in bounds]

    B2 = np.array(parts).sum()
    #now we add back the hard core correction, mayer_f goes to -1 in this limit
    if core_radius!=None:
        B2+=quad(hsolid,0,2*core_radius)[0]/2

    return B2, integrand, parts


def coord(frame, cutoff = (1.44635/1.4)*0.5*(1+np.sqrt(3))):
    """
    given a frame of 3D coordinates, calculates the coordination number of each
    particle using a simple cutoff radius. Also returns a list of particle neighbors who lie within this radius
    'shell radius': The cutoff radius used to find neighbors (defaulting to
    just beyond the first coordination shell of a close-packed lattice)
    author: Jack Bond
    """
    npart = frame.shape[0]
    i,j = np.mgrid[0:npart,0:npart]
    dr_norm = sp.spatial.distance.squareform(pdist(frame))
    
    neighbors = dr_norm<cutoff
    neighbors[i==j]=False
    Nc = np.sum(neighbors,axis=-1)

    return Nc, neighbors


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


def vor_coord(frame, flat=None, R=None, tol=1e-5, box_basis=None, exclude_border=False):
    """
    given a frame, calculates the coordination number of each particle using a voronoi tesselation.
    'flat': determine whether the frame in question sits on a spherical
    surface or not, will detect for itself if unspecified.
    'R': specify a radius for an underlying spherical surface. Will calculate
    if unspecified
    'tol': The tolerance passed to SphericalVoronoi. Basically it's how far
    from the spherical surface each particle is allowed to be at maximum.
    'box_basis': specify the basis vectors for period boundary conditions
    'exclude_border': for particles confined to a cap on a spherical surface,
    can opt to disregard (-1 coord number) particles who sit at the border of
    this cap and thus aren't part of a bulk system.
    author: Jack Bond
    """

    pnum, _ = frame.shape
    if flat is None:
        flat = (np.std(np.linalg.norm(frame,axis=-1)) > 0.1)
    
    if flat:
        if box_basis is None: raise Exception("please input simulation box")
        sv = Voronoi(expand_around_pbc(frame,box_basis)[:,:2])
        Vc = np.array([len(sv.regions[i]) for i in sv.point_region[:pnum]])
    
    else:
        minZ = min(frame[:,2])
        if R == None:
            radius = np.mean(np.linalg.norm(frame,axis=1))
        else:
            radius = R
        sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
    
        Vc = np.zeros(pnum)
        for i, region in enumerate(sv.regions):
            Vc[i] = len(region)
            if(exclude_border):
                for v in sv.vertices[region]:
                    if(v[2]<minZ):
                        Vc[i]+=-1
                        #Vc[i]=-1
    return Vc


def vor_coord_with_areas(frame, flat=None, R=None, tol=1e-5, box_basis=None, exclude_border=False):
    """
    given a frame, calculates the coordination number of each particle using a voronoi tesselation. Also returns the areas assigned to each voronoi cell
    'flat': determine whether the frame in question sits on a spherical
    surface or not, will detect for itself if unspecified.
    'R': specify a radius for an underlying spherical surface. Will calculate
    if unspecified
    'tol': The tolerance passed to SphericalVoronoi. Basically it's how far
    from the spherical surface each particle is allowed to be at maximum.
    'box_basis': specify the basis vectors for period boundary conditions
    'exclude_border': for particles confined to a cap on a spherical surface,
    can opt to disregard (-1 coord number) particles who sit at the border of
    this cap and thus aren't part of a bulk system.
    author: Jack Bond
    """

    pnum, _ = frame.shape
    if flat is None:
        flat = (np.std(np.linalg.norm(frame,axis=-1)) > 0.1)
    
    if flat:
        if box_basis is None: raise Exception("please input simulation box")
        sv = Voronoi(expand_around_pbc(frame,box_basis)[:,:2])
        Vc = np.array([len(sv.regions[i]) for i in sv.point_region[:pnum]])

        areas = np.zeros(pnum)
        for i,reg in enumerate(vc.point_region[:pnum]):
            indices = vc.regions[reg]
            if -1 in indices:
                areas[i] = np.infty
                print("this should never print")
            else:
                areas[i] = ConvexHull(vor.vertices[indices]).volume
    
    else:
        minZ = min(frame[:,2])
        if R == None:
            radius = np.mean(np.linalg.norm(frame,axis=1))
        else:
            radius = R
        sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
    
        Vc = np.zeros(pnum)
        areas = sv.calculate_areas()
        for i, region in enumerate(sv.regions):
            Vc[i] = len(region)
            if(exclude_border):
                for v in sv.vertices[region]:
                    if(v[2]<minZ):
                        areas[i] = np.infty
                        Vc[i]=-1
    return Vc, areas


def coordination_shell_densities(frame, flat=None, R=None, tol=1e-5, box_basis=None, exclude_border=False, coord_shell=(1.44635/1.4)*0.5*(1+np.sqrt(3))):
    """
    Given a frame, calclates the point-density based on the area of voronoi
    polygons INCLUDING NEAREST NEIGHBORS. Determines nearest neighbors via a 
    simple cutoff radius. Returns the local density per particle, the total
    area of of each particle's coordination shell, and the number of particles
    therein.
    author: Jack Bond
    """

    pnum, _ = frame.shape

    Vc, areas = vor_coord_with_areas(frame, flat=flat,R=R,tol=tol,box_basis=box_basis,exclude_border=exclude_border)
    _, neighbors = coord(frame, cutoff=coord_shell)

    shell_areas = 0*areas
    shell_counts = 0*Vc

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


#SCAR METHODS DO NOT WORK FOR FLAT SYSTEMS ATM
def find_charge_clusters(frame,tol=1e-5):
    """
    given a frame, finds and links together clusters of neighboring
    topological defects (non-6-coordinated particles). Returns an list of
    arrays of indices of these linked clusters, as well as an array of the net
    topological charges of the cluster.
    author: Jack Bond
    """
    if np.std(np.linalg.norm(frame,axis=-1)) > 0.1:
        raise Exception("input frame is not on a spherical surface")
    N = frame.shape[0]
    radius = np.mean(np.linalg.norm(frame,axis=1))
    sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
    qs = np.array([6-len(region) for region in sv.regions])

    shared_vertices = np.array([[np.sum(np.isin(r1,r2)) for r1 in sv.regions] for r2 in sv.regions])
    charged_pairs = np.abs(np.array([qs for _ in qs])*np.array([qs for _ in qs]).T)

    charged_neighbors = np.array((shared_vertices>=1)*(charged_pairs>0))

    clusts = []

    for i, ptcl in enumerate(charged_neighbors):
        links = np.sort(np.where(ptcl!=0)[0])
        if links.size !=0:
            unique_ptcls = True
            for j, clust in enumerate(clusts):
                if(np.any(np.isin(clust, links))):
                    new_clust = np.unique(np.append(clust,links))
                    clusts.pop(j)
                    clusts.append(new_clust)
                    links = new_clust
                    unique_ptcls = False
            if(unique_ptcls):
                clusts.append(links)

    net_charges = np.array([np.sum(qs[clust]) for clust in clusts])

    return clusts, net_charges
    #see older commits for a slightly simpler version of this method


def cluster_labels(frame,tol=1e-5):
    """
    given a frame, labels each particle with the net charge of the cluster to which it belongs, as well as the number of particles in that cluster
    """
    Sq = np.array([None for _ in range(frame.shape[0])])
    Sn = np.array([0 for _ in range(frame.shape[0])])

    clusts, net_charges = find_charge_clusters(frame,tol=tol)

    for i,c in enumerate(clusts):
        Sq[c] = net_charges[i]
        Sn[c] = len(c)

    return Sq, Sn


#CURRENTLY DEPRECATED, NEED TO REVISIT
def Psi6(frame, reference = np.array([0,1,0]),coord_shell=(1.44635/1.4)*0.5*(1+np.sqrt(3))):
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
def C6(frame,coord_shell=(1.44635/1.4)*0.5*(1+np.sqrt(3))):
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

    n, vc = coord(frame,cutoff=coord_shell)
    
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

