#!/usr/bin/env python
# coding: utf-8
# @author = Alex Yeh

import matplotlib.pyplot as plt
import FileHandling as handle

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R
import pandas as pd

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from itertools import product

def sector(angle, index):
    """Returns points in triangle defined by [1, 0, 0] and 
    [np.cos(rad), np.sin(rad), 0] with index points along each side."""
    rad = angle * (np.pi/180)
    a = np.array([1, 0, 0])
    b = np.array([np.cos(rad), np.sin(rad), 0])
    pairs = [pair for pair in product(range(index+1), repeat=2) if sum(pair) <= index]
    sector = []
    for i, j in pairs:
        tot = i*a + j*b
        sector.append(tot)
    return np.array(sector)

def patch(index):
    """returns n x 3 array of points composing a 60-120 rhombus in the xy-plane
    with edge # of points on each side. It is positioned with one corner at the
    origin and the rhombus in the +x, -y quarter plane."""
    rad = 60 * (np.pi/180) # 60 deg in radian
    a = np.array([1, 0, 0])
    b = np.array([np.cos(rad), np.sin(rad), 0])
    c = np.array([-np.cos(rad), np.sin(rad), 0])
    pairs = [p for p in product(range(index+1), repeat=2) if sum(p) <= index]
    rhomb = []
    for i, j in pairs:
        rhomb.append(i*a + j*b)
        if j > 0: #build lower half of patch
            rhomb.append(i*b + j*c)
    return np.array(rhomb)

def hk_face(h, k, e_tol=1e-3):
    """returns a h, k face within: (-0.5,0,0), (0,sqrt(3)/2,0), (0.5,0,0)
    This is ready to be ingested by slerp code"""

    if k == 0: # simplest case, default to simplest method
        return (sector(60, h)/h) - np.array([0.5, 0, 0])
    
    rad = 60 * (np.pi/180) # 60 deg in radian
    a = np.array([           1,           0, 0]) # first cardinal dir in lattice
    b = np.array([ np.cos(rad), np.sin(rad), 0]) # next cardinal direction
    c = np.array([-np.cos(rad), np.sin(rad), 0]) # helpful to complete triangles
    
    ray, top = h*a+k*b, h*b+k*c  # two rays which define triangle
    large_patch = rotate(patch(h+k), ray) # patch from which final will be cut
    
    # rotate rays to final orientation
    rot_ray, rot_top = rotate(ray, ray), rotate(top, ray)
    
    #define 1/slopes of each side of triangle
    #helps to cast problem in terms of x intercepts
    right_run_over_rise = (rot_top[0]-rot_ray[0])/(rot_top[1]-rot_ray[1])
    left_run_over_rise = rot_top[0]/rot_top[1]
    
    # inequalities to cut larger patch down to size
    right_mask = large_patch[:,0] <= rot_ray[0] + right_run_over_rise*large_patch[:,1] + e_tol
    left_mask = left_run_over_rise*large_patch[:,1] <= large_patch[:,0] + e_tol 
    x_axis_mask = large_patch[:,1]>=0
    
    # apply inequalities 
    trim = large_patch[left_mask & x_axis_mask & right_mask]
    
    # normalize triangle to edge length of 1 and shift to center on y axis
    normed_centered = (trim/np.linalg.norm(ray)) - np.array([0.5, 0, 0])
    return normed_centered

def rotate(points, ray):
    """Rotates the given array of points (n x 3) such that the projection of 
    ray (1 x 2 or 3) on the x-y plane would be brought to the x axis"""
    rad  = np.arctan2(ray[1], ray[0])
    rot  = np.array([[np.cos(rad), -np.sin(rad), 0],
                     [np.sin(rad),  np.cos(rad), 0],
                     [          0,            0, 1]])
    return points @ rot

# code adapted from:
# https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid
# barycentric coords for 2D triangle (-0.5,0) (0.5,0) (0,sqrt(3)/2)
def barycentricCoords(p):
    """Given 2D coordinate, returns 3D barycentric interpolation"""
    x,y = p
    # l3*sqrt(3)/2 = y
    l3 = y*2./np.sqrt(3.)
    # l1 + l2 + l3 = 1
    # 0.5*(l2 - l1) = x
    l2 = x + 0.5*(1 - l3)
    l1 = 1 - l2 - l3
    return np.array([l1,l2,l3])

def slerp(p0,p1,t):
    """Uniform interpolation of arc defined by p0, p1 (around origin)
       t=0 -> p0, t=1 -> p1"""
    assert abs(p0.dot(p0) - p1.dot(p1)) < 1e-7
    ang0Cos = p0.dot(p1)/p0.dot(p0)
    ang0Sin = np.sqrt(1 - ang0Cos*ang0Cos)
    ang0 = np.arctan2(ang0Sin,ang0Cos)
    l0 = np.sin((1-t)*ang0)
    l1 = np.sin(t    *ang0)
    return np.array([(l0*p0[i] + l1*p1[i])/ang0Sin for i in range(len(p0))])

# map 2D point p to spherical triangle s1,s2,s3 (3D vectors of equal length)
def mapPointToTriangle(p,s1,s2,s3):
    l1,l2,l3 = barycentricCoords(p)
    if abs(l3-1) < 1e-10: return s3
    l2s = l2/(l1+l2)
    p12 = slerp(s1,s2,l2s)
    return slerp(p12,s3,l3)

def mapFaceToTriangle(covering, triangle):
    """takes array of points and returns each point mapped to given triangle"""
    answer = []
    for point in covering:
        answer.append(mapPointToTriangle(point[:2], *triangle))
    return np.array(answer)

def mapFaceToIco(covering, icoTriangles):
    """takes in array of points in barycentric coordinates and maps to every
    face of icosahedron"""
    answer = []
    for tri in icoTriangles:
        answer.append(mapFaceToTriangle(covering, tri))
    multi_dim = np.reshape(np.array(answer), (-1,3))
    uniques = np.unique(multi_dim.round(decimals=10), axis=0)
    return uniques

#%% set up code for icosahedron adapted from:
# https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid
#upper half is z-axis, and 5 points on circle radius s at height h
h = 1/np.sqrt(5)
s = 2*h
topPoints = ([(0,0,1)]
             + [(s*np.cos(i*2*np.pi/5.),
                 s*np.sin(i*2*np.pi/5.),
                 h) for i in range(5)])

# bottom half is reflected across z and alternates with above points
bottomPoints = [(-x,y,-z) for (x,y,z) in topPoints]
icoPoints = np.array(topPoints + bottomPoints)
# reorder triangles to always increase in counter-clockwise direction
icoTriPoints = np.array([[ 0,  1,  2],
                         [ 0,  2,  3],
                         [ 0,  3,  4],
                         [ 0,  4,  5],
                         [ 0,  5,  1],
                         [ 6,  7,  8],
                         [ 6,  8,  9],
                         [ 6,  9, 10],
                         [ 6, 10, 11],
                         [ 6, 11,  7],
                         [ 1,  9,  2],
                         [ 2,  8,  3],
                         [ 3,  7,  4],
                         [ 4, 11,  5],
                         [ 5, 10,  1],
                         [ 1, 10,  9],
                         [ 2,  9,  8],
                         [ 3,  8,  7],
                         [ 4,  7, 11],
                         [ 5, 11, 10]])
icoTri = np.array([[icoPoints[p] for p in icoTriPoints[i]] for i in range(len(icoTriPoints))])

def hk_sphere(h, k, e_tol=1e-3):
    triangle = hk_face(h, k, e_tol)
    return mapFaceToIco(triangle, icoTri)

if __name__=="__main__":
    #%% troubleshooting h k patches
   
    def get_tri_edges(h, k):
        theta = 60 * (np.pi/180)
        a = np.array([1, 0, 0])
        b = np.array([np.cos(theta), np.sin(theta), 0])
        c = np.array([-np.cos(theta), np.sin(theta), 0])
        return np.array([0*a, h*a+k*b, h*b+k*c, 0*a])
    
    equivalent_shells = [(5,3), (7,0)]
    large_patch = patch(max([h+k for h,k in equivalent_shells]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.scatter(large_patch[:,0], large_patch[:,1], alpha=0.25)
    for h, k in equivalent_shells:
        curr = get_tri_edges(h, k)
        ax.plot(curr[:,0], curr[:,1], label=f'({h},{k})')
    ax.legend()
    ax.set_title('T=49 faces cut from a triangular lattice')
    fig.savefig("cutting_faces.jpg", bbox_inches='tight')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for h, k in equivalent_shells:
        curr = hk_face(h, k)
        print(f"{h}, {k}: {curr.shape[0]}")
        ax.scatter(curr[:,0], curr[:,1], 
                   label=f'({h},{k}) N:{curr.shape[0]}',alpha=0.5)
    ax.legend()
    ax.set_title('T=49 faces and the total number of particles in each triangle')
    ax.set_xlim([-0.6, 0.6])
    fig.savefig("T49_overlaid.jpg", bbox_inches='tight')
    
    
    #%% cycling the face corners
    
    
    #%% h, k over initial range, with some important metrics
    temp_stats = {}
    for h in range(1, 10):
        for k in range(0, h+1):
            t = h**2 + h*k + k**2
            curr_sphere = hk_sphere(h, k)
            min_spacing = pdist(curr_sphere).min()
            min_r = 1/min_spacing  #minimum radius so particles are 1 unit apart
            assert 10*t + 2 == curr_sphere.shape[0], 'hk_sphere does not match analytic expression'
            pnum = 10*t + 2
            # below finds eta_eff if R is in units of a_eff
            eta_eff = (pnum*0.5**2)/(4*min_r**2)
            temp_stats[(h, k)] = (h, k, t, min_spacing, min_r, pnum, eta_eff)
            
    hk_stats = pd.DataFrame.from_dict(temp_stats, orient='index',
                                      columns=('h', 'k', 'tnum', 'min_spacing', 'min_r', 'pnum', 'eta_eff'))
    
    #%% plotting clearly
    fig, ax = plt.subplots()
    for k in np.unique(hk_stats.k):
        mask = hk_stats.k == k
        ax.scatter(hk_stats[mask].pnum, hk_stats[mask].eta_eff,
                   label=f"(h, {k})")
    ax.legend()
    ax.set_xlabel('number of particles')
    ax.set_ylabel(r'$\eta_{eff}$', size=12)
    ax.set_title('$\eta_{eff}$ of icosahedral shells with particles at $2a_{eff}$')
    fig.savefig("eta_eff_over_pnum_k_colored.jpg", bbox_inches='tight')

    fig, ax = plt.subplots()
    for h in np.unique(hk_stats.h):
        mask = hk_stats.h == h
        ax.scatter(hk_stats[mask].pnum, hk_stats[mask].eta_eff,
                   label=f"({h}, k)")
    ax.legend()
    ax.set_xlabel('number of particles')
    ax.set_ylabel(r'$\eta_{eff}$', size=12)
    ax.set_title('$\eta_{eff}$ of icosahedral shells with particles at $2a_{eff}$')
    fig.savefig("eta_eff_over_pnum_h_colored.jpg", bbox_inches='tight')
    
    fig, ax = plt.subplots()
    for k in np.unique(hk_stats.k):
        mask = hk_stats.k == k
        ax.scatter(hk_stats[mask].pnum, hk_stats[mask].min_r,
                   label=f"(h, {k})")
    ax.legend()
    ax.set_xlabel('number of particles')
    ax.set_ylabel(r'minimum radius [$2a_{eff}$]', size=12)
    ax.set_title(r'Radius of icosahedral shells with particles at $2a_{eff}$')
    fig.savefig("rmin_over_pnum_k_colored.jpg", bbox_inches='tight')
