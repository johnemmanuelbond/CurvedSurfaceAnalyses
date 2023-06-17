import os, sys, glob, json

from FileHandling import read_xyz_frame

import numpy as np
import scipy as sp

from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi, Voronoi#, geometric_slerp
from scipy.signal import find_peaks
from scipy.integrate import quad

import matplotlib as mpl
import matplotlib.pyplot as plt

from timeit import default_timer as timer

"""from a a spherical cap in 3D coordinates, returns a 2D projection
in polar coordinates where the radius is the arclength from the pole and
the angle is the azimuthal angle. Also returns said coordinates in a
cartesian representation, and the jacobian at each point for considering length
scales"""
def cap_polar_projection(frame):
    Rs = np.linalg.norm(frame,axis=-1)
    l = Rs*np.arccos(frame[:,2]/Rs)
    phi = np.arctan(frame[:,1]/(frame[:,0]+0.000001))+np.pi*(frame[:,0]<0)
    x,y = l*np.cos(phi), l*np.sin(phi)
    jacobian = Rs*np.sin(l/Rs)
    return np.array([x,y]).T, np.array([l,phi]).T, jacobian


#a simple coordination number 
def Nc(frame, shellradius = (1.44635/1.4)*0.5*(1+np.sqrt(3))):
    npart = frame.shape[0]
    i,j = np.mgrid[0:npart,0:npart]
    dr_norm = sp.spatial.distance.squareform(pdist(frame))
    
    neighbors = dr_norm<shellradius
    neighbors[i==j]=False
    Nc = np.sum(neighbors,axis=-1)

    return Nc, neighbors

#coordination number based of voronoi triangulation
def Vc(frame,excludeborder=False,R=None,tol=1e-6,flat=False, box_basis=None):
    pnum, _ = frame.shape
    
    if flat:
        expand = lambda frame, basis: np.array([
            *frame,
            *(frame+basis@np.array([1,0,0])),
            *(frame+basis@np.array([-1,0,0])),
            *(frame+basis@np.array([0,1,0])),
            *(frame+basis@np.array([0,-1,0])),
            *(frame+basis@np.array([1,1,0])),
            *(frame+basis@np.array([-1,1,0])),
            *(frame+basis@np.array([1,-1,0])),
            *(frame+basis@np.array([-1,-1,0])),
            ])

        sv = Voronoi(expand(frame,box_basis))
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
            if(excludeborder):
                for v in sv.vertices[region]:
                    if(v[2]<minZ):
                        Vc[i]+=-1
                        #Vc[i]=-1
    return Vc

def shareVoronoiVertex(sv, i, j):
    vertices_i = sv.regions[i]
    vertices_j = sv.regions[j]

#returns an Nx3 array of rgb values based on the voronoi tesselation of a frame
def voronoi_colors(frame,v=None,tol=1e-6):
    if type(v)==type(None):
        v = Vc(frame, excludeborder=False,tol=tol)
    #print(np.sum(6-v))
    #print(np.sum(np.abs(6-v)))
    colors = np.array([[0.6,0.6,0.6] for _ in v])
    greens = np.array([[0,0.5*vi/6,0.2] for vi in v])
    reds = np.array([[1-0.5*vi/6,0,0.2+0] for vi in v])
    colors[v>6] = greens[v>6]
    colors[v<6] = reds[v<6]
    return colors

#point-density based on the area of voronoi polygons on a frame
def rho_voronoi(frame,excludeborder=False,R=None,tol=1e-6,flat=False):
    minZ = min(frame[:,2])
    
    if flat:
        sv = Voronoi(frame)
    else:
        if R == None:
            radius = np.mean(np.linalg.norm(frame,axis=1))
        else:
            radius = R
        sv = SphericalVoronoi(frame, radius = radius,threshold=tol)

    V_rho = np.zeros(frame.shape[0])
    for i, area in enumerate(sv.calculate_areas()):
        V_rho[i] = 1/area
    return V_rho

#point-density based on the area of voronoi polygons INCLUDING NEAREST NEIGHBORS on a frame
def rho_voronoi_shell(frame,excludeborder=False,R=None,tol=1e-6, flat=False,coord_shell=(1.44635/1.4)*0.5*(1+np.sqrt(3))):
    minZ = min(frame[:,2])
    
    if flat:
        sv = Voronoi(frame)
    else:
        if R == None:
            radius = np.mean(np.linalg.norm(frame,axis=1))
        else:
            radius = R
        sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
    

    _, neighbors = Nc(frame,shellradius=coord_shell)

    V_rho = np.zeros(frame.shape[0])
    areas = sv.calculate_areas()
    for i, nei in enumerate(neighbors):
        #Nc does not include the particle itself when counting neighbors
        As = areas[[i,*np.where(nei!=0)[0]]]
        area = As.sum()
        nshell = As.size
        V_rho[i] = nshell/area
    return V_rho

#returns an Nx3 array of rgb values based on the voronoi tesselation of a frame
def density_colors(frame,rhos=None,aeff = 0.5,tol=1e-6):
    if type(rhos) == type(None):
        rhos = rho_voronoi(frame, excludeborder=False,tol=tol)
    #print(np.sum(6-v))
    #print(np.sum(np.abs(6-v)))
    rho_cp = 0.9067/(np.pi*aeff**2)
    rho_fl = 0.69/(np.pi*aeff**2)
    rho_mean = rhos.mean()
    scale = (rhos-rho_mean)/(rho_cp-rho_fl)
    colors = np.array([[0.5-s,0.5+s,0.5] for s in scale])
    return colors

 
"""
Given a spherical frame, plots the centers and vertices of the voronoi tesselation
source: plot_voronoi, 5/26/23
author: John Edison
"""
def plot_voronoi_sphere(points, filename="voronoi_frame.jpg", colors = ['yellow','yellow','yellow','yellow','orange','red','grey','green','blue','purple'], view = np.array([0,0,1]),tol=1e-5):

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as a3d
    
    radius = np.linalg.norm(points,axis=-1).mean()
    center = np.array([0.0, 0.0, 0.0])
    sv     = SphericalVoronoi(points, radius, center,threshold=tol)
    sv.sort_vertices_of_regions()
    areas = sv.calculate_areas()

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
    
    for i, region in enumerate(sv.regions):

        if (areas[i] > 1.50*np.pi):
            continue
        
        n = len(region)
        vtx = sv.vertices[region]
        
        tri = a3d.art3d.Poly3DCollection([vtx])

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

"""
Given a planar frame, plots the centers and vertices of the voronoi tesselation
source: plot_voronoi, 5/26/23
author: John Edison, Jack Bond
"""
def plot_voronoi_plane(points, filename="voronoi_frame.jpg", colors = ['yellow','yellow','yellow','yellow','orange','red','grey','green','blue','purple'],box=None):

    if box is None: raise Exception('please input simulation box')
    pnum,_= points.shape
    expand = lambda frame, basis: np.array([
            *frame,
            *(frame+basis@np.array([1,0,0])),
            *(frame+basis@np.array([-1,0,0])),
            *(frame+basis@np.array([0,1,0])),
            *(frame+basis@np.array([0,-1,0])),
            *(frame+basis@np.array([1,1,0])),
            *(frame+basis@np.array([-1,1,0])),
            *(frame+basis@np.array([1,-1,0])),
            *(frame+basis@np.array([-1,-1,0])),
            ])
    vor = Voronoi(expand(points, box)[:,:2])

    fig = plt.figure(figsize=(7,7),dpi=600)
    ax = fig.add_subplot()

    plt.axis('off')

    for i, pr in enumerate(vor.point_region[:pnum]):
        region = vor.regions[pr]
        n = len(region)
        vtx = vor.vertices[region]
        
        if -1 in region:
            continue

        tri = mpl.collections.PolyCollection([vtx])

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
    
#scar methods DO NOT WORK for flat systems
def findScarsCarefully(frame,tol=1e-6):
    N = frame.shape[0]
    radius = np.mean(np.linalg.norm(frame,axis=1))
    sv = SphericalVoronoi(frame, radius = radius,threshold=tol)
    qs = np.array([6-len(region) for region in sv.regions])

    shared_vertices = np.array([[np.sum(np.isin(r1,r2)) for r1 in sv.regions] for r2 in sv.regions])
    charged_pairs = np.abs(np.array([qs for _ in qs])*np.array([qs for _ in qs]).T)

    charged_neighbors = np.array((shared_vertices>=1)*(charged_pairs>0))

    scars = []

    for i, ptcl in enumerate(charged_neighbors):
        links = np.sort(np.where(ptcl!=0)[0])
        if links.size !=0:
            uniquePtcls = True
            for j, scar in enumerate(scars):
                if(np.any(np.isin(scar, links))):
                    newscar = np.unique(np.append(scar,links))
                    scars.pop(j)
                    scars.append(newscar)
                    links = newscar
                    uniquePtcls = False
            if(uniquePtcls):
                scars.append(links)

    scarCharges = np.array([np.sum(qs[scar]) for scar in scars])

    return scars, scarCharges


def findScars(frame,tol=1e-6,coord_shell=(1.44635/1.4)*0.5*(1+np.sqrt(3))):

    _, neighbors = Nc(frame,shellradius=coord_shell)
    charge = 6-Vc(frame,tol=tol)
    
    charged_pairs = np.abs(np.array([charge for _ in charge])*np.array([charge for _ in charge]).T)

    charged_neighbors = np.array(neighbors*charged_pairs)

    scars = []
    
    # pairs = np.unique(np.sort(np.array(np.where(charged_neighbors)).T,axis=-1),axis=0)
    # for pair in pairs:
    #   uniquePtcls = True
    #   for i, scar in enumerate(scars):
    #       if(np.any(np.isin(scar, pair))):
    #           newscar = np.unique(np.append(scar,pair))
    #           scars.pop(i)
    #           scars.append(newscar)
    #           uniquePtcls = False
    #   if(uniquePtcls):
    #       scars.append(pair)
    
    for i, ptcl in enumerate(charged_neighbors):
        links = np.sort(np.where(ptcl!=0)[0])
        if links.size !=0:
            uniquePtcls = True
            for j, scar in enumerate(scars):
                if(np.any(np.isin(scar, links))):
                    newscar = np.unique(np.append(scar,links))
                    scars.pop(j)
                    scars.append(newscar)
                    links = newscar
                    uniquePtcls = False
            if(uniquePtcls):
                scars.append(links)

    scarCharges = np.array([np.sum(charge[scar]) for scar in scars])

    return scars, scarCharges

def ScarNumber(frame,tol=1e-6):
    Sc = np.array([None for _ in range(frame.shape[0])])

    scars, scarCharges = findScars(frame,tol=tol)

    for i, scar in enumerate(scars):
        Sc[scar] = scarCharges[i]

    return Sc

#returns an Nx3 array of rgb values based on the net voronoi charge of a frame
def scar_colors(frame,s=None,tol=1e-6):
    if type(s) == type(None):
        s = ScarNumber(frame,tol=tol)
    colors = np.zeros((s.size,3))
    for i,si in enumerate(s):
        if si == None:
            colors[i] = np.array([0.6,0.6,0.6])
        elif si == 0:
            colors[i] = np.array([0.2,0.2,0.6])
        elif si > 0:
            colors[i] = np.array([0.5+(si-1)/2,0.2,0.2])
        elif si < 0:
            colors[i] = np.array([0.2,0.5+(1-si)/2,0.2])

    return colors


def shells(pnum):
    """from particle number, calculate number of shells assuming hexagonal crystal"""
    # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
    return -1/2 + np.sqrt((pnum-1)/3 + 1/4)

def c6_hex(pnum):
    """returns C6 for a hexagonal cluster of the same size"""
    # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
    s = shells(pnum)
    return 6*(3*s**2 + s)/pnum

def Psi6(frame, reference = np.array([0,1,0]),coord_shell=(1.44635/1.4)*0.5*(1+np.sqrt(3))):
    N = frame.shape[0]
    i,j = np.mgrid[0:N,0:N]
    _, n = Nc(frame,shellradius=coord_shell)
    vc = Vc(frame)


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
    N = frame.shape[0]
    n, vc = Nc(frame,shell_radius=coord_shell)
    
    psi6, psi6global = Psi6(frame)
    C6 = 0*vc

    for k, v in enumerate(vc):
        pi = psi6[k]
        pj = np.conjugate(psi6[n[k]==1])
        chi = np.abs(np.real(pi*pj))/np.abs(pi*pj)
        C6[k] = np.sum(chi>=0.32)

    return C6, np.mean(C6)/c6_hex(N), psi6, psi6global

from scipy.special import gamma
# Second virial coeffecient for an arbitrary potential
# recieve a pair potential in kT units, phi, which takes r in simulation units
# returns the second virial coefficient for that potential
def B2(phi,splits=np.array([0,5,np.infty]),dim=3, core_radius=None):
    
    if type(phi) != type(np.mean): raise Exception("Input must be a python function")

    mayer_f = lambda r: np.exp(-phi(r))-1

    #hypersphere solid angle
    g = gamma(dim/2+1)
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

if __name__=="__main__":

    nargs = len(sys.argv)
    if nargs <= 1:
        nfiles = len(glob.glob("output*.xyz"))
        index = np.arange(nfiles-1) 
        c = np.zeros(nfiles-1)
        p = np.zeros(nfiles-1)
        for i in index:
            frame = np.array(read_xyz_frame(f"output_{i+1}.xyz"))
            _, c[i], _, p[i] = C6(frame)

        fig, ax = plt.subplots()
        
        simArgument = json.load(open("simArgument.json",'r'))
        R = simArgument['radius']

        ax.set_title(f"Order Parameters for R = {R:.3f}")
        ax.set_ylim([0,1])
        ax.set_xlabel("sweeps")
        ax.scatter(index*simArgument['nsnap'],c, label = r"[$C_6$]")
        ax.scatter(index*simArgument['nsnap'],p, label = r"[$\psi_6$]")
        ax.legend()
        plt.savefig("Order Parameters",bbox_inches = 'tight')

    elif nargs == 2:

        from Correlation import firstCoordinationShell
        framepath = sys.argv[1]
        frame = np.array(read_xyz_frame(framepath))
        coord_shell = firstCoordinationShell(frame)
        nc, n = Nc(frame,shellradius=coord_shell)
        vc = Vc(frame)
        
        rdf,mids,_ = g_r(frame)

        a = mids[np.argmax(rdf[mids<coord_shell])]
        #print(nc[np.where(vc!=nc)],vc[np.where(nc!=vc)], (6-nc[np.where(vc!=nc)])-(6-vc[np.where(nc!=vc)]))

        print(f"Voronoi Tesselation: total charge: {np.sum(6-vc)} excess charge: {0.5*(np.sum(np.abs(6-vc))/12-1)}")
        print(f"Coordination Number: total charge: {np.sum(6-nc)} excess charge: {0.5*(np.sum(np.abs(6-nc))/12-1)}")

        start = timer()
        qs = 6-nc
        end = timer()
        print(f"charges computed in {end-start}s")


        print(f"First Coordination Shell: {coord_shell}")
        print(f"R/a: {np.mean(np.linalg.norm(frame,axis=-1))/a}")
        fig, ax = plt.subplots()
        ax.plot(rs,hval,lw=0.6,c="black")
        plt.show()

        findScars(frame)

        # simArgument = json.load(open("simArgument.json",'r'))
        # R = simArgument['radius']

        # proj, _,_ = capPolarProjection(frame)
        # fig,[ax1,ax2] = plt.subplots(1,2, figsize = (8,4))

        # plt.suptitle(f"Projected snapshot R={R:.3f}")

        # plt.tight_layout()#rect=[0, 0.03, 1, 1.2])
        # ax1.set_aspect('equal', 'box')
        # ax1.set_title("Particles Colored by Voronoi Charge")
        # ax1.scatter(proj[:,0],proj[:,1],c=nc)


        # localC, meanC, psi, globalPsi = C6(frame)
        # print(meanC,globalPsi)

        # ax2.set_aspect('equal', 'box')
        # ax2.set_title(r"Particles Colored by $C_6$")
        # ax2.scatter(proj[:,0],proj[:,1],c=localC,cmap='viridis')
        # plt.savefig(f"{framepath}.png",bbox_inches = 'tight')

    elif nargs > 2:
        raise Exception("You entered too much stuff, fool")


# #for planar systems only
# def Ci6(coordinates, shellradius = 1.6):
#   npart = coordinates.shape[0]
#   i,j = np.mgrid[0:npart,0:npart]
#   #dr = np.sqrt((coords[p,0]-coords[q,0])**2+(coords[p,1]-coords[q,1])**2)
#   dr_vec = coordinates[i]-coordinates[j]
#   dr_norm = np.linalg.norm(dr_vec,axis=-1)
#   dr_norm[i==j] = 1e-3
    
#   #to get the theta's we need an aribrtrary reference angle, we'll say (0,1) is our reference angle
#   # we get the cosine with the dot product between dr and (0,1), and then divide it by the norm
#   cosines = dr_vec[:,:,0]/dr_norm
#   #fix nans:
#   cosines[i==j]=0
    
#   neighbors = dr_norm<shellradius
#   neighbors[i==j]=False
#   Nc = np.sum(neighbors,axis=-1)
#   Nc[Nc==0]=1
    
#   with np.errstate(divide='ignore'):

#       argument = np.exp(6j * np.arccos(cosines))

#       psi6 = np.array([(1/Nc[n])*np.sum(neighbors[n]*argument[n]) for n in range(npart)])

#       psi6star = np.conjugate(psi6)

#       chi6 = np.abs(np.real(psi6[i]*psi6star[j]))/np.abs(psi6[i]*psi6star[j])

#       C6 = np.array([(1/6)*np.sum(neighbors[n]*(chi6[n]>0.32)) for n in range(npart)])

#   return C6, Nc

def Fourier_Transform_2D(coordinates,pointnumber=100,filename="Fourier_Test_2D.png"):
    N = coordinates.shape[0]
    coordinates[coordinates==0]=0.000001
    voronoiNumber = Vc(coordinates)

    Rs = np.linalg.norm(coordinates,axis=-1)
    l = Rs*np.arccos(coordinates[:,2]/Rs)
    phi = np.arctan(coordinates[:,1]/coordinates[:,0])+np.pi*(coordinates[:,0]<0)
    x,y = l*np.cos(phi), l*np.sin(phi)
    
    R = np.mean(Rs)
    h = (N*(np.pi*0.5**2)/0.65)/(2*np.pi*R)
    extent = R*np.arccos((R-h)/R)
    if np.isnan(extent):
        extent = np.pi*R

    fig,[ax1,ax2] = plt.subplots(1,2, figsize = (8,4))
    plt.tight_layout()#rect=[0, 0.03, 1, 1.2])
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim([-1*extent,extent])
    ax1.set_ylim([-1*extent,extent])
    ax2.set_aspect('equal', 'box')

    plt.suptitle(f"Projected snapshot and Fourier Transform for R={R:.2f}[2a]")
    ax1.set_xlabel("[2a]")
    ax1.set_ylabel("[2a]")

    ax1.scatter(x,y,color=[getRGB(v) for v in voronoiNumber])
    #ax.scatter(coordinates[:,0],coordinates[:,1],marker = 'x')
    

    mesh = np.linspace(-1*np.pi,np.pi,num=pointnumber,endpoint=True)
    kx, ky = np.meshgrid(mesh,mesh)
    
    #mesh = np.linspace(0,np.pi,num=pointnumber,endpoint=True)
    #kl, kphi = np.meshgrid(mesh,2*mesh)
    #kx, ky = kl*np.cos(kphi), kl*np.sin(kphi)

    jacobian = Rs*np.sin(l/Rs)
    dot = np.einsum("i,jk,i->ijk",jacobian,kx,np.cos(phi)) + np.einsum("i,jk,i->ijk",jacobian,ky,np.sin(phi))
    #dot = np.einsum("i,jk,i,jk->ijk",jacobian,kl,np.cos(phi),np.cos(kphi)) + np.einsum("i,jk,i,jk->ijk",jacobian,kl,np.sin(phi),np.sin(kphi))

    integrand = np.einsum('ijk,i->ijk',np.exp(-2j*np.pi*dot),jacobian)
    F = 1/(2*np.pi)*np.sum(integrand,axis=0)

    #ax = plt.axes(projection='3d')
    sigma = np.std(np.abs(F))
    zval = np.abs(F)
    #zval[zval<sigma]=0
    zval[zval>4*sigma]=4*sigma
    ax2.contourf(kx,ky,zval,cmap='Greys')
    #ax2.contourf(kl,kphi,zval,cmap='Greys')


    # ax2.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    # ax2.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    # ax2.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    # ax2.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))

    fig.savefig(filename, bbox_inches='tight')
    plt.close()
