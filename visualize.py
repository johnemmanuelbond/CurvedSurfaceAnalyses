# -*- coding: utf-8 -*-
"""
Created on Mon, Feb 28, 2022

Preps and opens Monte Carlo files for viewing in ovito with colors according to
a specific order parameter.

@author: Jack Bond
"""

import numpy as np
import scipy as sp

import os, glob, sys, json

import OrderParameters as order
import UnitConversions as units
import FileHandling as handle
import DriverUtilities as util

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from timeit import default_timer as timer

ovito = "\"C:\\Program Files\\OVITO Basic\\ovito.exe\""

from MCBatchAnalyzer import sample_frames, firstCoordinationShell, pair_charge_correlation, scar_correlation, BoltzmannInversion

if __name__=="__main__":
    framepath = ""

    r_ico = np.sin(2*np.pi/5)
    theta1 = 2*np.arcsin(1/2/r_ico)
    theta2 = 2*np.arccos((r_ico**2+r_ico**2*np.cos(theta1/2)**2-3/4)/(2*r_ico**2*np.cos(theta1/2)))

    fig,ax = plt.subplots()
    ax.set_xlabel(r"Geodesic Distance $[2a]$")
    #ax.set_ylim([0,2])
    ax.set_xlim([0,5])
    ax.set_ylabel(r"$g(r)$")
    #ax.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
    #ax.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')

    fig55,ax55 = plt.subplots()
    ax55.set_xlabel(r"Geodesic Distance [rad/$\pi$]")
    ax55.set_ylim([0,2])
    ax55.set_xlim([0,1])
    ax55.set_ylabel(r"$g_{{5-5}}$")
    ax55.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--')
    ax55.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')
    
    figScar,axScar = plt.subplots()
    axScar.set_xlabel(r"Geodesic Distance [rad/$\pi$]")
    axScar.set_ylim([0,2])
    axScar.set_xlim([0,1])
    axScar.set_ylabel(r"$g_{{scar-scar}}$")
    axScar.axvline(x=theta1/np.pi,ymax=2,lw=0.6,c="black")#,label=r"$\theta_{{1}}$",ls='--',")
    axScar.axvline(x=theta2/np.pi,ymax=2,lw=0.6,c="red")#,label=r"$\theta_{{2}}$",ls='--')

    figCharge, axCharge = plt.subplots()
    axCharge.set_title("Excess Charge vs Sweeps")
    axCharge.set_ylabel(r"$\frac{{1}}{{2}}(\frac{{\sum|q_{{i}}|}}{{12}}-1)$")
    axCharge.set_xlabel("sweeps")

    nargs = len(sys.argv)
    if nargs <= 1:
        simFolder = os.getcwd()

        config = json.load(open(simFolder + "/configFile.json",'r'))
        simArgument = config['simargument']
        params = config['params']
        inter = config['interactions'][0]
        a = params['particle_radius']
        aeff = units.getAEff(params)
        R = simArgument['radius']
        N = simArgument['npart']

        eta_eff = np.round(N*(aeff/(2*a))**2/(4*R**2),3)

        sim = sample_frames([simFolder+"/"],label="movie_voronoi",last_section=1.0,reset=True)
        col = np.array([order.voronoi_colors(frame) for frame in sim])
        handle.output_vis("movie_voronoi.atom",sim,colors=col)
        col = np.array([order.density_colors(frame,aeff=aeff/(2*a)) for frame in sim])
        handle.output_vis("movie_density.atom",sim,colors=col)

        frames = sample_frames([simFolder+"/"],label="samples",last_section=1/2,reset=True)
        # initFrame = handle.read_xyz_frame("output_0.xyz")
        # _, info = order.radialDistributionFunction(initFrame)
        # spacing = info['particle_spacing']

        vals, mids, bins = order.g_r(frames,shell_radius=R,bin_width=0.01)
        ax.plot(mids,vals,lw=0.5)

        max_gr_idx = np.argmax(vals[mids<1.5])
        spacing = mids[max_gr_idx]
        ax.set_ylim([0,1.1*vals[max_gr_idx]])
        fig.savefig("g(r).jpg")

        vcs = np.array([order.Vc(frame, R = R) for frame in frames])
        rhos = np.array([order.rho_voronoi(frame, R = R) for frame in frames])
        excessVC = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)

        output = {
                "R": R,
                "a_gr": spacing,
                "aeff": aeff/a,
                "N": N,
                "qXS_bar": excessVC.mean(axis=-1),
                "qXS_std": np.std(excessVC),
                "rho_bar": rhos.mean(),
                "rho_std": np.std(rhos),
                "eta_eff": eta_eff,
                "eta_bar": (np.pi*(aeff/(2*a))**2)*rhos.mean(),#((1/rhos).mean())**-1,
        }

        handle.dumpDictionaryJSON(output, "./values")

        ax55.set_title(rf"$\eta_{{eff}}$ = {eta_eff}, R/a = {R/spacing:.2f}")
        axScar.set_title(rf"$\eta_{{eff}}$ = {eta_eff}, R/a = {R/spacing:.2f}")

        midss55 = []
        gs55 = []
        midssScar = []
        gsScar = []

        for frame in frames:
            #R = np.mean(np.linalg.norm(frame,axis=-1))

            mids55, g55, _, _ = pair_charge_correlation(1,1,frame,R,bin_width=R*np.pi/40)
            midss55.append(mids55/(np.pi*R))
            gs55.append(g55)

            midsScar, gScar, _, _ = scar_correlation(frame,R,bin_width=R*np.pi/40)
            midssScar.append(midsScar/(np.pi*R))
            gsScar.append(gScar)

        mids55 = np.mean(np.array(midss55),axis=0)
        g55 = np.mean(np.array(gs55),axis=0)
        midsScar = np.mean(np.array(midssScar),axis=0)
        gScar = np.mean(np.array(gsScar),axis=0)



        ax55.plot(mids55, g55,lw = 0.5)
        #plt.show()
        fig55.savefig("5-5 Pair Correlation.jpg", bbox_inches='tight')

        axScar.plot(midsScar, gScar,lw = 0.5)
        #plt.show()
        figScar.savefig("Scar-Scar Pair Correlation.jpg", bbox_inches='tight')
            
        sweeps = simArgument['nsnap']*(1+np.arange(sim.shape[0]))
        vcs = [order.Vc(frame, R = R) for frame in sim]

        excessVC = 0.5*(np.array([np.sum(np.abs(6-vc)) for vc in vcs])/12-1)
        axCharge.plot(sweeps,excessVC,lw=0.6)

        figCharge.savefig("Excess Charge per Sweep")

    elif nargs == 2:
        framepath = sys.argv[1]
        frame = np.array(read_xyz_frame(framepath))

        R = np.mean(np.linalg.norm(frame,axis=-1))

        #coordinationNumber = order.Nc(frame,shellradius = order.firstCoordinationShell(frame))
        voronoiNumber = order.Vc(frame,excludeborder=False)

        print(order.firstCoordinationShell(frame))

        #diff = ((6-coordinationNumber)-(6-voronoiNumber))
        #uninteresting = (coordinationNumber==6)*(voronoiNumber == 6) 

        orderParameter = voronoiNumber#+5*diff

        relevant = orderParameter>0#frame[:,2]>0

        charge = 6-orderParameter

        excessCharge = 0.5*(np.sum(np.abs(charge))/12-1)
        print(f"Total Charge {np.sum(charge)}\nExcess Charge: {excessCharge}")

        start = timer()
        scars, scarCharges = order.findScars(frame)
        end = timer()
        print(f"{end-start}s scarfinding time")
        scarindex = np.zeros((len(orderParameter),4))
        for i,scar in enumerate(scars):
            scarindex[scar] = cm.hsv(i/len(scars))
            # if(scarCharges[i]==0):
            #   scarindex[scar] = np.array([2,0,0,0])

        vFile = open("visual.xyz", "w")
        vFile.write(f"{orderParameter[relevant].size+1}\n")
        vFile.write(f"{orderParameter.size} total particles snapshot\n")
        vFile.write(f"{R-0.5} 0.0 0.0 0.0 0.8, 0.8, 0.9\n")
        for i, N in enumerate(orderParameter[relevant]):
            vFile.write("0.5 ")
            for c in frame[relevant][i]:
                vFile.write(f"{c} ")
            r,g,b,_ = scarindex[i]#getRGB(N)
            vFile.write(f"{r} {g} {b}\n")

        vFile.close()
        os.system(f"{ovito} visual.xyz")

        mids55, g55, _, _ = pair_charge_correlation(1,1,frame,R,bin_width=R*np.pi/20)
        mids55*=1/(np.pi*R)

        ax55.plot(mids55, g55,lw = 0.5)
        #plt.show()
        fig55.savefig("5-5 Pair Correlation.jpg", bbox_inches='tight')

        midsScar, gScar, _, _ = scar_correlation(frame,R,bin_width=R*np.pi/20)
        midsScar*=1/(np.pi*R)

        axScar.plot(midsScar, gScar,lw = 0.5)
        #plt.show()
        figScar.savefig("Scar-Scar Pair Correlation.jpg", bbox_inches='tight')


    elif nargs > 2:
        raise Exception("You entered too much stuff, fool")

    
