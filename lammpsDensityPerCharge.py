# -*- coding: utf-8 -*-
"""
Created on Sat, Oct 22, 2022

uses Alex Yeh's lammps analyss tools  to correlate local density (via voronoi areas) to
topological charge (via voronoi vertices)

@author: Jack Bond
"""
import numpy as np
from scipy import integrate
import os, sys, json, glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from timeit import default_timer as timer


import main_lib.FileHandling as handle
from main_lib.UnitConversions import getAEff
from main_lib.OrderParameters import Vc, rho_voronoi, rho_voronoi_shell
from main_lib.Correlation import firstCoordinationShell

hexatic_window = 0.702,0.714

fig, [ax,axshell] = plt.subplots(1,2,sharex=True,sharey=True)
ax.set_title("Single Particle")
ax.set_xlabel("Topological Charge")
ax.set_ylabel(r"$\eta_{eff}$")
ax.axhline(y=hexatic_window[0],color='red',lw=0.6,ls='--')
ax.axhline(y=hexatic_window[1],color='blue',lw=0.6,ls='--')

axshell.set_title("First Coord Shell")
#axshell.set_xlabel("Topological Charge")
#axshell.set_ylabel(r"$\eta_{eff}$")
axshell.axhline(y=hexatic_window[0],color='red',lw=0.6,ls='--')
axshell.axhline(y=hexatic_window[1],color='blue',lw=0.6,ls='--')

single = len(sys.argv) == 1 or sys.argv[1] != 'batch'
if len(sys.argv)>=2: batch = sys.argv[1]=='batch'

if single:
    # load data
    start = timer()
    path = os.getcwd()+"/"
    print(path)
    #%% get run info
    infile = glob.glob(path+'*.in')
    assert len(infile) == 1, "need to have one specified input file"

    with open('config.json','r') as c:
        config = json.load(c)
        params = config['params']
        simarg = config['arg']
    multiple = np.load(path+'datapts.npy')
    fnum, N, _ = multiple.shape
    coordination = np.load(path+'vor_coord.npy')

    aeff = getAEff(params)/(2*params['particle_radius'])
    R = simarg['xxxradiusxxx']

    eta_eff = N*(aeff**2)/(4*R**2)
    ax.set_ylim([eta_eff-0.1,eta_eff+0.1])

    #loop through equilibrated frames to gather information about charged clusters. We don't want to excessively calculate cluster positions, and we only want to sample independent frames, so I've created an array of index positions to control which frames we actually perform calculations on.
    last_section = 1/3
    desired_samples = 100
    idx = np.arange(int((1-last_section)*fnum),fnum,int((last_section*fnum)/desired_samples))

    lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
    pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

    frames = multiple[idx]
    shell = firstCoordinationShell(frames)

    qs = np.array([6-Vc(frame, R = R,tol=1e-5) for frame in frames]).flatten()
    #XS = 0.5*(np.array([np.sum(np.abs(q)) for q in qs])/12-1)
    etas = np.array([rho_voronoi(frame,R=R,tol=1e-5) for frame in frames]).flatten()*np.pi*(aeff**2)
    etashells = np.array([rho_voronoi_shell(frame,R=R,tol=1e-5,coord_shell=shell) for frame in frames]).flatten()*np.pi*(aeff**2)

    pltqs=[]
    pltetas=[]
    pltetashells = []
    pltdetas=[]
    pltdetashells = []

    for q in np.unique(qs):
        eta = etas[qs==q].mean()
        etashell = etashells[qs==q].mean()
        counts = etas[qs==q].size
        print(f"q:{q}, counts:{counts}")
        deta = etas[qs==q].std()
        detashell = etashells[qs==q].std()

        pltqs.append(q)
        pltetas.append(eta)
        pltetashells.append(etashell)
        pltdetas.append(deta)
        pltdetashells.append(detashell)

    ax.errorbar(pltqs,pltetas,yerr=pltdetas,label=pltlab, ls='none', marker='^',fillstyle='none')
    axshell.errorbar(pltqs,pltetashells,yerr=pltdetashells,label=pltlab, ls='none', marker='^',fillstyle='none')

    handle.dumpDictionaryJSON({'qs':pltqs,'etas':pltetas,'detas':pltdetas,'etashells':pltetashells,'detashells':pltdetashells,'eta_eff':eta_eff,'R':R,'N':N},"ChargeAndDensity")

elif batch:
    outs = glob.glob("./BD_R*eta*/ChargeAndDensity.json")
    for o in outs:

        dic = json.load(open(o,'r'))
        eta_eff, R, N = dic['eta_eff'], dic['R'], dic['N']

        if eta_eff>0.68:

            lab = f"eta_eff={eta_eff:.3f},R={R:.2f},N={N}"
            pltlab = rf"$\eta_{{eff}}$={eta_eff:.3f},R={R:.2f},N={N}"

            pltqs= dic['qs']
            pltetas= dic['etas']
            pltetashells = dic['etashells']
            pltdetas = dic['detas']
            pltdetashells = dic['detashells']

            ax.set_ylim([0.5,1])
            ax.errorbar(pltqs,pltetas,yerr=pltdetas,label=pltlab, ls='none', marker='^',fillstyle='none')
            axshell.errorbar(pltqs,pltetashells,yerr=pltdetashells,label=pltlab, ls='none', marker='^',fillstyle='none')

axshell.legend(loc='center right', bbox_to_anchor=(2.2, 0.5))

fig.savefig("DensityChargeCorrelation.jpg",bbox_inches='tight')

