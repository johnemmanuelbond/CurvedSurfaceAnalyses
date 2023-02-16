# -*- coding: utf-8 -*-
"""
Created on Sat, Sep 10, 2022

Plots the mean squared displacement, and several variations thereof for a lammps run

Also performs fits to find the diffusivities

@author: Jack Bond, Alex Yeh
"""

import glob, os, sys, json

import numpy as np
from numpy.random import default_rng
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from main_lib.FileHandling import read_infile, read_dump, read_thermo, get_thermo_time, dumpDictionaryJSON, output_vis
from main_lib.UnitConversions import kb, getAEff
from main_lib.Correlation import theta1,theta2
from main_lib.MSD import *


if __name__=="__main__":

    from timeit import default_timer as timer

    allstart = timer()

    # load data
    path = './'
    print(path)
    #%% read dump file
    reset = False

    '''reading infiles'''

    start = timer()
    infile = glob.glob(path+'*.in')
    assert len(infile) == 1, "need to have one specified input file"

    lammps_params = read_infile(infile[0])
    thermo_header, thermo = read_thermo(path+'log.lammps')
    time_str = get_thermo_time(path+'log.lammps')

    filename = path+'out.dump'
    if os.path.exists(path+'datapts.npy'):
        multiple = np.load(path+'datapts.npy')
        ts = np.load(path+'times.npy')
    else:
        multiple, ts = read_dump(filename)
        np.save(path+'datapts.npy',multiple)
        np.save(path+'times.npy',ts)

    config = json.load(open('config.json','r'))
    params = config['params']

    a_hc = params['particle_radius']
    a_eff = getAEff(params)
    kT = params['temperature']*kb
    visc = params['viscosity']
    D_SI = kT/(6*np.pi*visc*a_hc)

    fnum, pnum, _ = multiple.shape
    # dt = lammps_params['timestep']*tau # [s]

    damp = lammps_params['damp']
    mass = config['arg']['xxxmassxxx']
    temp = config['arg']['xxxtempxxx']
    D0 = temp*damp/mass

    tau_D = 1/(4*D0)
    tau_D_SI = (a_hc**2)/(D_SI)

    #kappa = lammps_params['kappa_2a']/(2*a_hc)
    #bpp = lammps_params['bpp']
    dt = lammps_params['timestep']

    times = ts*dt

    if lammps_params['rad'] is None:
        shell_radii = np.linalg.norm(multiple,axis=-1).mean(axis=-1)
        shell_radius = shell_radii.min() #get stand in value for rough approximation
        all_eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radii**2)
        eta_eff = all_eta_eff.max() #get stand in value for rough approximation
        title = (f"N: {pnum}, "
                 +f"R: {shell_radii.max():0.3f}-{shell_radii.min():0.3f} "
                 +r"[2a], $\eta_{eff}$:" 
                 + f"{all_eta_eff.max():0.3f}-{all_eta_eff.min():0.3f}")
    else:
        shell_radius = lammps_params['rad']
        eta_eff = (pnum*(a_eff/(2*a_hc))**2)/(4*shell_radius**2)
        title = (f"N: {pnum}, "
                 +f"R: {shell_radius:0.3f} "
                 +r"[2a], $\eta_{eff}$:" 
                 + f"{eta_eff:0.3f}")

    end = timer()
    interval = end - start
    print(f"read and process files {interval:.2f}s")

    #%% obtaining theoretical diffusivities
    start = timer()
    taus = thermo[:,0]-thermo[:,0].min()
    all_taus = np.linspace(0, thermo[:,0].max(), num=150)
    theo = sphere_msd(all_taus, D0, shell_radius)

    msd_func = lambda x, D0: sphere_msd(x, D0, shell_radius=shell_radius)
        
    totmsd_coef, totmsd_cov = curve_fit(msd_func, taus, thermo[:,-1], p0=[1e-3])
        
    #%% calculate msd
    msd_time_scale = min(2000,int(times.shape[0]/2))

    s = 100 - 50*(pnum<300)- 25*(pnum<50) #- 15*(pnum<10)

    #recall we sometimes one frozen particle now, so we simply disregard the frozen particle when calculating displacements, use multiple[:,1:,:] to do that
    msd_comp = mto_msd(multiple, msd_time_scale,skips = s)
    msd_part = mto_msd_part(multiple, msd_time_scale, skips = s)
    msd = msd_comp.sum(axis=-1)
    msd_times = times[:msd_time_scale]
    #np.savetxt(path+f'msd_{msd_time_scale}frames.txt', (msd_times, msd), header='tau msd[2a]^2')

    #%% get bootstrap error
    trials = 1000
    rng = default_rng()
        
    #get confidence intervals
    msd_ci = bootstrap_mto_msd(msd_part, trials, rng=rng)

    diff_coef, diff_cov = curve_fit(msd_func, msd_times, msd, p0=[1e-1])
    theo = sphere_msd(msd_times, D0, shell_radius)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(msd_times, msd, label='mto msd')
    ax.fill_between(msd_times, msd-msd_ci[0], msd+msd_ci[1],
                    alpha=0.3, label='95% bootstrap ci')
    ax.plot(msd_times, theo, color='k', ls=':', label=f'D={D0:0.1e}')
    ax.plot(msd_times, msd_func(msd_times, *diff_coef), 
            color='C0', ls='-.',
            label=f'D={diff_coef[0]:0.3f} (fit)')

    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, msd_times[-1]])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, min(1.1*msd_func(msd_times[-1], *diff_coef),1.2*2*shell_radius**2)])

    ax.plot(msd_times, np.ones(msd_times.size)*2*shell_radius**2,ls='-',label=r'$2R^2$')
    #ax.plot(msd_times, np.ones(msd_times.size)*np.pi*shell_radius**2,ls='-',label=r'$\pi R^2$')

    ax.set_title(title)

    ax.legend()
    fig.savefig(path+"msd.jpg", bbox_inches='tight')
    # ax.plot(thermo[:msd_time_scale,0], thermo[:msd_time_scale,-1], label='lammps msd')
    # fig.savefig(path+"mto_msd_comparison.jpg", bbox_inches='tight')

    short_time = 5*damp
    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, short_time])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, 1.1*4*D0*short_time])
    ax.set_title(title)

    ax.legend()
    fig.savefig(path+"msd_damp.jpg", bbox_inches='tight')

    short_time = 0.03
    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, short_time])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, 1.1*4*D0*short_time])
    ax.set_title(title)

    ax.legend()
    fig.savefig(path+"msd_short.jpg", bbox_inches='tight')

    ### Trying to do a center of mass trick to eliminate lattice diffusion from our plots. To do this we lock onto a subtended sector of particles and then perform mto msd on that subset. If the lattice is diffusing back and forth, this should pick that up.

    sector_centers = shell_radius*np.array([[0,0,1],[0,1,0]])#,[])
    subtended_angles = np.array([theta1,theta1])#,np.pi/5,np.pi,1.5*np.pi)

    for i, (center, sub_ang) in enumerate(zip(sector_centers,subtended_angles)):

        center[0] = max(center[0],1e-7)
        
        phi = np.arctan(center[1]/center[0]) + np.pi*(center[0]<0)
        theta = np.arccos(center[2]/np.linalg.norm(center))

        _,_,_,subset,_ = sector_msd(multiple,theta_c = theta, phi_c = phi,subtended_halfangle=sub_ang/2)
        movie_rate = 4
        movie = multiple[np.arange(fnum)%movie_rate==0]
        output_vis(f"sector_{i}.atom",movie[:,subset,:])

        msd_com_3, msd_ens_3, md_rad, mean_n, c_vec = mto_sector_msd(multiple,msd_time_scale,skips=s, theta_c = theta, phi_c = phi,subtended_halfangle=sub_ang/2)

        msd_com = msd_com_3.sum(axis=-1)
        msd_ens = msd_ens_3.sum(axis=-1)
        
        fig,ax=plt.subplots(figsize=(5,5))
        ax.plot(msd_times,msd, label="Full ensemble mto msd",lw=0.8)
        ax.plot(msd_times,msd_ens, label="Subset ensemble mto msd")
        ax.plot(msd_times,msd_com, label="Subset C.O.M. mto msd")
        ax2 = ax1.twinx()
        ax2.plot(msd_times,msd_rad, label="Subset C.O.M. mean radial disp.",lw=0.8)

        ax.set_xlabel("[$\\tau$]", fontsize=12)
        ax.set_xlim([0, msd_times[-1]])
        ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
        ax2.set_ylabel("[$\sigma$]", fontsize=12)
        ax.set_ylim([0, min(1.1*msd_func(msd_times[-1], *diff_coef),1.2*2*shell_radius**2)])
        ax.set_title(title + f"\n Sector: {np.round(c_vec,2)}, $\theta_{{sub}}$={2*sub_ang/np.pi:.2f}$\pi$ rad, $N_s$~{mean_n:.1f}")

        ax.legend()
        fig.savefig(path+f"msd_sector_{i}.jpg", bbox_inches='tight')

    # ax.set_title(f"{title}\nPinned Particle at {np.round(pin,2)}")
    # ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    # fig.savefig(path+"msd_com_location.jpg", bbox_inches='tight')

    #### For viewing several sectors on the same plot

    #sometimes we pin particles, heres how to acount for that
    #locate the pinned particle

    # pin = multiple[0,0,:]
    # phi_c_1 = np.arctan(pin[1]/pin[0]) + np.pi*(pin[0]<0)
    # theta_c_1 = np.arccos(pin[2]/np.linalg.norm(pin))

    # thetas = np.array([theta_c_1, theta_c_1 + np.pi/4, theta_c_1 + np.pi/2]) % np.pi

    #first we vary the location of the subtended sector

    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.plot(msd_times, msd, label='ensemble mto msd')
    
    # for i, theta in enumerate(thetas):
    #     msd_com, msd_rad, mean_n, c_vec = mto_com_sector_msd(multiple,msd_time_scale,skips=s, theta_c = theta, phi_c = phi_c_1)
        
    #     ax.plot(msd_times,msd_com.sum(axis=-1), label = f"com msd about {np.round(c_vec,2)}\n(~{mean_n:.1f} ptcls)", color=f"C{i+1}",lw=0.8)
    #     ax.plot(msd_times,msd_rad, label = f"radial com msd about {np.round(c_vec,2)}", color=f"C{i+1}",lw=1.1, ls="-.")

    # ax.set_xlabel("[$\\tau$]", fontsize=12)
    # ax.set_xlim([0, msd_times[-1]])
    # ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    # ax.set_ylim([0, min(1.1*msd_func(msd_times[-1], *diff_coef),1.2*2*shell_radius**2)])

    # ax.set_title(f"{title}\nPinned Particle at {np.round(pin,2)}")
    # ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    # fig.savefig(path+"msd_com_location.jpg", bbox_inches='tight')

    # #Now we vary the size of the subtended sector

    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.plot(msd_times, msd, label='ensemble mto msd')
    
    # subtends = np.array([np.pi/10,theta1/2,np.pi/4,theta2/2,np.pi/2])

    # for i, subtend in enumerate(subtends):
    #     msd_com, msd_rad, mean_n, c_vec = mto_com_sector_msd(multiple,msd_time_scale,skips=s, theta_c = (theta_c_1+np.pi/2) % np.pi, phi_c = phi_c_1,subtended_halfangle = subtend)
        
    #     ax.plot(msd_times,msd_com.sum(axis=-1), label = f"com msd about {np.round(c_vec,2)}\nsubtended angle: {2*subtend/np.pi:.2f}$\pi$ rad", color=f"C{i+1}",lw=0.8)
    #     ax.plot(msd_times,msd_rad, label = f"radial com msd about {np.round(c_vec,2)}\n(~{mean_n:.1f} ptcls)", color=f"C{i+1}",lw=0.6, ls="-.")

    # ax.set_xlabel("[$\\tau$]", fontsize=12)
    # ax.set_xlim([0, msd_times[-1]])
    # ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    # ax.set_ylim([0, min(1.1*msd_func(msd_times[-1], *diff_coef),1.2*2*shell_radius**2)])

    # ax.set_title(f"{title}\nPinned Particle at {np.round(pin,2)}")
    # ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    # fig.savefig(path+"msd_com_size.jpg", bbox_inches='tight')


    #if the voronoi tesselation is already done we'll do the charge-weighted msd too
    if os.path.exists(path+'vor_coord.npy'):
        
        coordination = np.load(path+'vor_coord.npy')
        msd5,msd6,msd7 = mto_msd_part_Vcweight(multiple[:,1:,:], coordination[:,1:], msd_time_scale,skips=s)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
        ax.set_xlabel("[$\\tau$]", fontsize=12)
        ax.plot(msd_times, msd, label='overall', color='k', zorder=5)
        ax.plot(msd_times, msd5, label='5', color='red',lw=0.6,ls='--')
        ax.plot(msd_times, msd6, label='6', color='gray',lw=0.6,ls='--')
        ax.plot(msd_times, msd7, label='7', color='green',lw=0.6,ls='--')
        ax.set_title(title)
        ax.legend()


        # frac=0.9
        # msdhex,nhex = mto_msd_hex(multiple[:,1:,:], coordination[:,1:], msd_time_scale,skips=s,min_six_frac=frac)

        # fig, ax = plt.subplots(figsize=(5,5))
        # ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
        # ax.set_xlabel("[$\\tau$]", fontsize=12)
        # ax.plot(msd_times, msd, label='overall', color='k', zorder=5,lw=0.6)
        # ax.plot(msd_times, msdhex, label='mostly 6-fold', color='blue',lw=0.8)
        # axn = ax.twinx()
        # axn.set_ylabel("Number of Applicable Particles")
        # axn.plot(msd_times,nhex, color='blue',lw=0.6,ls='--')
        # ax.set_title(f"MSD for particles which spend {100*frac:.1f}% as 6-coordinated")
        # ax.legend()

        # fig.savefig(f"./msd_local.jpg", bbox_inches='tight')

    end = timer()
    print(f"msd calculation {end - start:.2f}s")

    config = json.load(open('config.json', 'r'))

    output = {
            'D_0_fit': diff_coef[0],
            'D_0': D0,
            'D_SI': D_SI,
            'a_hc_SI': a_hc,
            'tau_SI': tau_D_SI/tau_D,
            'D_SI_conv': diff_coef[0]*(2*a_hc)**2/(tau_D_SI/tau_D)
    }

    dumpDictionaryJSON(output, 'diffusion')

    allend = timer()
    print(f"full runtime {allend-allstart:.2f}s")
