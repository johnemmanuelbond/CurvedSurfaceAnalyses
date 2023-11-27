# -*- coding: utf-8 -*-
"""
Created on Fri, Jun 23, 2023

Pulled several methods scattered about other files into a document with my most commonly
used geometry formulas and values.

@author: Jack Bond
"""

import numpy as np

#icosohedral angles
r_ico = np.sin(2*np.pi/5)
theta1 = 2*np.arcsin(1/2/r_ico)
theta2 = 2*np.arccos((r_ico**2+r_ico**2*np.cos(theta1/2)**2-3/4)/(2*r_ico**2*np.cos(theta1/2)))


def rho(eta, a_eff):
    """converts area fraction to number density based on an effective radius"""
    return eta/(np.pi * a_eff**2)  #in units of 1/[a_eff**2]


def eta(rho, a_eff):
    """converts number density to area fraction based on an effective radius"""
    return rho*(np.pi * a_eff**2) #unitless


def eta_eff_from_N_R(N,R,a_eff):
    """
    Given a particle count and a sphere radius, returns effective area fraction on such a sphere
    Assumes R and a_eff given in 2a units
    author: Jack Bond
    """
    return N * a_eff**2 / (4 * R**2)


def R_from_eta_eff_N(eta_eff,N,a_eff):
    """
    Given an area fraction and a particle count, returns the radius of sphere on which those particles would pack
    Assumes R and a_eff given in 2a units
    author: Jack Bond
    """
    return np.sqrt(N * a_eff**2 / (4 * eta_eff))


def N_from_eta_eff_R(eta_eff,R,a_eff):
    """
    Given an area fraction and a sphere radius, returns the number of particles which would fit on that sphere, rounded to the nearest integer
    Assumes R and a_eff given in 2a units
    author: Jack Bond
    """
    return np.round(eta_eff * 4 * R**2/(a_eff**2))


def chord_to_arc(dists, radius):
    """
    converts a chord accross a sphere with radius 'radius' into a geodesic arc
    source: MSD 6/20/23
    author: Jack Bond
    """
    args = dists/(2*shell_radius)
    args[args>1]=1
    arcs = 2*shell_radius*np.arcsin(args)
    return arcs


def hoomd_box_to_matrix(box):
    """ returns the matrix form of a hoomd box for use in minimum image calculations"""
    return np.array([[box[0],box[3]*box[1],box[4]*box[2]],[0,box[1],box[5]*box[2]],[0,0,box[2]]])


def hoomd_matrix_to_box(box):
    """ returns the hoomd box form of a pbc basis matrix"""
    hbox= np.array([box[0,0],box[1,1],box[2,2],box[0,1]/box[1,1],box[0,2]/box[2,2],box[1,2]/box[2,2]])
    if box[2,2]==0:
        hbox[4]=0
        hbox[5]=0
    return hbox


def expand_around_pbc(frame, basis, do_corners = False):
    """
    given a frame and a box basis matrix, returns a larger frame which
    including surrounding particles from the nearest images. This will
    enable scipy.voronoi to respect periodic boundary conditions
    author: Jack Bond
    """
    adjusted = np.array([
                *frame,
                *(frame+basis@np.array([1,0,0])),
                *(frame+basis@np.array([-1,0,0])),
                *(frame+basis@np.array([0,1,0])),
                *(frame+basis@np.array([0,-1,0])),])

    if do_corners:
        adjusted =  np.array([
                    *adjusted,
                    *(frame+basis@np.array([1,1,0])),
                    *(frame+basis@np.array([-1,1,0])),
                    *(frame+basis@np.array([1,-1,0])),
                    *(frame+basis@np.array([-1,-1,0])),
                    ])

    return adjusted


def sphere_msd(taus, damp, shell_radius = 10):
    """
    Theoretical expression for the diffusive mean-squared displacement on a
    spherical surface.
    source: general_analysis, 7/23/22
    author: Alex yeh
    """
    return 2*(shell_radius**2)*(1-np.exp(-2*damp*taus*shell_radius**-2))


def minimum_image(coords, wraps, basis):
    """
    uses the minumum image convention to correctly account for perodic
    boundary conditions when calculating coordinates by using the basis
    vectors of the periodic box.
    source MSD, 2/9/23
    author: Jack Bond
    """
    disp = np.einsum("ij,anj->ani",basis,wraps)
    return coords + disp


def cap_polar_projection(frame):
    """
    From a a spherical cap in 3D coordinates, returns a 2D projection
    in polar coordinates where the radius is the arclength from the pole and
    the angle is the azimuthal angle. Also returns said coordinates in a
    cartesian representation, and the jacobian at each point for considering length
    scales
    source = OrderParameters (prehistoric)
    author = Jack Bond
    """
    Rs = np.linalg.norm(frame,axis=-1)
    l = Rs*np.arccos(frame[:,2]/Rs)
    phi = np.arctan(frame[:,1]/(frame[:,0]+0.000001))+np.pi*(frame[:,0]<0)
    x,y = l*np.cos(phi), l*np.sin(phi)
    jacobian = Rs*np.sin(l/Rs)
    return np.array([x,y]).T, np.array([l,phi]).T, jacobian


#### DEPRECATED THING WHICH IS PERHAPS WORTH RESURRECTING
# def Fourier_Transform_2D(coordinates,pointnumber=100,filename="Fourier_Test_2D.png"):
#     N = coordinates.shape[0]
#     coordinates[coordinates==0]=0.000001
#     voronoiNumber = Vc(coordinates)

#     Rs = np.linalg.norm(coordinates,axis=-1)
#     l = Rs*np.arccos(coordinates[:,2]/Rs)
#     phi = np.arctan(coordinates[:,1]/coordinates[:,0])+np.pi*(coordinates[:,0]<0)
#     x,y = l*np.cos(phi), l*np.sin(phi)
    
#     R = np.mean(Rs)
#     h = (N*(np.pi*0.5**2)/0.65)/(2*np.pi*R)
#     extent = R*np.arccos((R-h)/R)
#     if np.isnan(extent):
#         extent = np.pi*R

#     fig,[ax1,ax2] = plt.subplots(1,2, figsize = (8,4))
#     plt.tight_layout()#rect=[0, 0.03, 1, 1.2])
#     ax1.set_aspect('equal', 'box')
#     ax1.set_xlim([-1*extent,extent])
#     ax1.set_ylim([-1*extent,extent])
#     ax2.set_aspect('equal', 'box')

#     plt.suptitle(f"Projected snapshot and Fourier Transform for R={R:.2f}[2a]")
#     ax1.set_xlabel("[2a]")
#     ax1.set_ylabel("[2a]")

#     ax1.scatter(x,y,color=[getRGB(v) for v in voronoiNumber])
#     #ax.scatter(coordinates[:,0],coordinates[:,1],marker = 'x')
    

#     mesh = np.linspace(-1*np.pi,np.pi,num=pointnumber,endpoint=True)
#     kx, ky = np.meshgrid(mesh,mesh)
    
#     #mesh = np.linspace(0,np.pi,num=pointnumber,endpoint=True)
#     #kl, kphi = np.meshgrid(mesh,2*mesh)
#     #kx, ky = kl*np.cos(kphi), kl*np.sin(kphi)

#     jacobian = Rs*np.sin(l/Rs)
#     dot = np.einsum("i,jk,i->ijk",jacobian,kx,np.cos(phi)) + np.einsum("i,jk,i->ijk",jacobian,ky,np.sin(phi))
#     #dot = np.einsum("i,jk,i,jk->ijk",jacobian,kl,np.cos(phi),np.cos(kphi)) + np.einsum("i,jk,i,jk->ijk",jacobian,kl,np.sin(phi),np.sin(kphi))

#     integrand = np.einsum('ijk,i->ijk',np.exp(-2j*np.pi*dot),jacobian)
#     F = 1/(2*np.pi)*np.sum(integrand,axis=0)

#     #ax = plt.axes(projection='3d')
#     sigma = np.std(np.abs(F))
#     zval = np.abs(F)
#     #zval[zval<sigma]=0
#     zval[zval>4*sigma]=4*sigma
#     ax2.contourf(kx,ky,zval,cmap='Greys')
#     #ax2.contourf(kl,kphi,zval,cmap='Greys')


#     # ax2.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
#     # ax2.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
#     # ax2.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
#     # ax2.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))

#     fig.savefig(filename, bbox_inches='tight')
#     plt.close()
