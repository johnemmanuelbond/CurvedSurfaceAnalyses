# -*- coding: utf-8 -*-
"""
Created on Sat, Jul 23, 2022

A collection of methods for reading/writing commonly used MC and lammps filetypes.
Collected from various driver and analysis scripts authored by myself and @AlexYeh

@author: Jack Bond
"""


import numpy as np
import json#, yaml
import gsd


def dump_json(dic,filename):
    """code for dumping the relevant dictionaries into files"""
    file = open(filename,"w")
    file.write(json.dumps(dic,indent=2))
    file.close()


# def dump_yaml(dic, filename):
#    file = open(filename,"w")
#    yaml.dump(dic,file,sort_keys=False)
#    file.close()


def read_xyz_frame(filename):
    """
    Reads xyz coordinates from a .xyz file. Expected format:
    number of particles
    comment line (should have units)
    type x-coord y-coord z-coord
    type x-coord y-coord z-coord
    Returns a triply nested numpy array, with format:
    [ # first frame
     [-10.34, -10.8, 37.1], #coordinates of particle
     [-14.48, 5.69, 36.85],
     [-7.47, -16.2, 35.8],
    source: MCBatchAnalyzer, 7/23/22
    author: Alex Yeh, Jack Bond
    """
    
    frame = []
    with open(filename, 'r') as xyzfile:
        pnum = None
        for i, line in enumerate(xyzfile):
            if i == 0:
                pnum = float(line.strip())
            elif i == 1:
                pass  #throw away comment
            elif i <= pnum+1:
                # assumes format as below for particle coordinates
                # index xcomp ycomp zcomp ... (unspecified afterwards)
                coordinates = line.split()[1:4]
                frame.append([float(coord) for coord in coordinates])
            else:
                print("extra: " + line)
    return np.array(frame)


def save_xyz(coords, filename, comment=None):
    """
    Given a frame or set of frames, saves them to filename as xyz file
    source: general_analysis, 7/23/22
    author: Alex yeh
    """
    
    if comment == None:
        comment = "idx x(um)   y(um)   z(um)   token\n"
    if len(coords.shape) == 2:
        coords = coords[np.newaxis,:] #make single frames correct size
    #print(filename)
    with open(filename, 'w', newline='') as output:        
        for i, frame in enumerate(coords):
            #print number of particles in frame
            output.write("{}\n".format(frame.shape[0]))
            output.write(comment)
            for j, part in enumerate(frame):
                output.write("{}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                             j+1,
                             *part))


def save_vis_xyz(frame, colors, filename="visual", comment="A colored visualization",show_shell=True):
    """
    given a Nx3 frame and an Nx3 array of rgb colors, outputs a pretty visualization for your viewing pleasure in Ovito.
    See OrderParameters.voronoi_colors for an example coloring.
    source: some driver, 8/10/22
    author: Jack Bond
    """
    
    N = frame.shape[0]
    R=np.linalg.norm(frame,axis=-1).mean()

    vFile = open(filename, "w")
    vFile.write(f"{N+show_shell} \n")
    vFile.write(f"{N} real particles {comment} \n")
    if show_shell: #can choose not to visualize the central spherical shell by setting the kwarg to false
        vFile.write(f"{R-0.5} 0.0 0.0 0.0 0.8, 0.8, 0.9 \n")
    for i in range(N):
        vFile.write("0.5 ")
        for c in frame[i]:
            vFile.write(f"{c} ")
        for c in colors[i]:
            vFile.write(f"{c} ")
        vFile.write("\n")
    vFile.close()


def output_vis(filename, frames, radii=None, ts=None, colors=None, alphas = None, box=None, show_shell = True):
    """
    outputs dump file in format to be used by OVITO for visualization
    frames are allowed to have different particle counts over time
    alphas and colors control the visuals per frame
    if show_shell, then a grey sphere will sit at 0,0 with a radius specified by radii
    source: general_analysis, 7/23/22
    author: Alex Yeh, Jack Bond
    """

    if show_shell and (radii is None): # calculate average radius at each frame
        radii = np.linalg.norm(frames,axis=-1).mean(axis=-1)
    
    if ts is None: # initialize timesteps if not provided
        ts = np.arange(len(frames), dtype=np.int16)
    
    if box is None: # initialize box if necessary
        box = format_boxes((-0.5,0.5), (-0.5,0.5), (-0.5,0.5))
    
    if colors is None: # initialize colors to light grey (0.6, 0.6, 0.6)
        colors = [np.ones_like(frame) * 0.6 for frame in frames]

    if alphas is None: # initialize alphas to completely opaque
        alphas = [np.zeros(frame.shape[0]) for frame in frames]

    pnum = frames[0].shape[0]    
    header = f"LAMMPS data file for {pnum} particles in a fluid\n\n{pnum} atoms\n1 atom types\n"
    mass = "Masses\n\n1 1.0\n"
    title = "\nAtoms\n\n"
    
    with open(filename, 'w') as outfile:
        for i, frame in enumerate(frames):
            pnum = frame.shape[0]
            col = colors[i]
            al = alphas[i]
            outfile.write(f"ITEM: TIMESTEP\n{ts[i]}\n")
            outfile.write(f"ITEM: NUMBER OF ATOMS\n{pnum+show_shell}\n") #add in sphere
            outfile.write(f"ITEM: BOX BOUNDS ff ff ff" + box)
            outfile.write(f"ITEM: ATOMS id radius xs ys zs color.r color.g color.b transparency\n")
            if show_shell: #Can choose not to print the spherical shell with the kwarg.
                outfile.write(f"0 {radii[i]-0.5:0.5f} 0 0 0 1.0 1.0 1.0, 0.0\n")
            for p, part in enumerate(frame):
                line = f"{p+1} 0.5 "
                coords = " ".join([f"{val:0.5f}" for val in part]) + " "
                coloring = " ".join([f"{val:0.5f}" for val in col[p]])
                alpha= f" {al[p]}"
                outfile.write(line+coords+coloring+alpha+'\n')
        outfile.close()


def format_boxes(xbox, ybox, zbox):
    """
    simple method to make a string for the simulation box needed
    to write a faux-lammps dump file.
    source: general_analysis, 7/23/22
    author: Alex Yeh
    """
    
    out = ("\n" +
           " ".join([str(i) for i in xbox]) + " xlo xhi\n" +
           " ".join([str(i) for i in ybox]) + " ylo yhi\n" +
           " ".join([str(i) for i in zbox]) + " zlo zhi\n")
    return out


def get_thermo_time(filename):
    """
    returns the total runtime of the lamps sim?
    source: general_analysis, 7/26/22
    author: Alex Yeh
    """

    with open(filename) as logfile:
        for line in logfile:
            if line.startswith('Total wall time:'):
                return line.split()[-1]
    # if no wall time, job was ended before calculation completed
    return 'overtime'


def read_thermo(filename):
    """
    Gets the thermodynamic quantites from a log.lammps file. Quantites like
    the energy, temperature, and cartesian msds per coordinate and as a whole
    per each timestep. The user specifies how lammps outputs this in the
    .in file.
    source: general_analysis, 7/23/22
    author: Alex Yeh
    """

    reading = False
    results = []
    with open(filename) as logfile:
        for line in logfile:
            if line.startswith('Loop time of'):
                reading = False
                
            if line.strip().startswith('Time'):
                reading = True
                header = line.split()
                ncol = len(header)
            elif reading:
                # break if there's an inconsistency with the line
                if len(line.split()) != ncol:
                    print("results did not end normally")
                    break
                row = [float(i) for i in line.split()]
                results.append(row)
    return header, np.array(results)


def read_infile(filename):
    """
    extracts experimental parameters from the file used to run a lammps sim (.in)
    Returns a dict of each value set in the calculation.
    source: general_analysis, 7/23/22
    author: Alex Yeh
    """

    starts = [['unit', 1, 'unit'],
              ['timestep', 1, 'timestep'],  # split idx for timestep
              ['fix temp all langevin', 6, 'damp'],  # split idx for damp
              ['fix temp most langevin', 6, 'damp'],  # split idx for damp
              ['pair_coeff 1 1', 3, 'bpp'], # [kT]
              ['pair_style yukawa/colloid', 2, 'kappa_2a'], # [1/(2a)]
              ['fix step all nve/manifold/rattle', -1, 'rad'], #get shell radius
              ]

    txt = []
    with open(filename) as infile:
        for line in infile:
            txt.append(line)
            
    txt_arr = np.array(txt)

    out = {}
    for pre, offset, name in starts:
        mask = np.char.startswith(txt_arr, pre)
        if np.any(mask):
            curr = txt_arr[mask][0]
            if pre == 'unit':
                out[name] = curr.split()[offset]
            elif pre == 'fix step all nve/manifold/rattle':
                end = curr.split()[-1]
                if end.startswith('v_'): #then we are varying radius
                    out[name] = None
                    continue
                # constant radius is saved
                out[name] = float(curr.split()[offset])
            else:
                out[name] = float(curr.split()[offset])

    if out:
        if 'timestep' not in out:
            out['timestep'] = 0.005
        return out
    else:
        raise ValueError(f'no valid lines in {filename}')


def read_dump(filename):
    """
    converts a lammps dump file into a numpy array of time-ordered frames.
    each frame contains the coordinates of each particle.
    source: general_analysis, 7/23/22
    author: Alex Yeh
    """
    time_acc = []
    pnum = None
    coord_acc = []

    with open(filename, 'r') as outfile:
        file_iter = enumerate(outfile)
        for i, line in file_iter:
            if line.startswith('ITEM: TIMESTEP'):
                i, line = next(file_iter)
                time_acc.append(int(line.strip()))
            # currently can only parse simulations with constant particle #
            if line.startswith('ITEM: NUMBER OF ATOMS') and pnum is None:
                i, line = next(file_iter)
                pnum = int(line.strip())
            if line.startswith('ITEM: ATOMS id type'):
                frame = []
                for n in range(pnum):
                    i, line = next(file_iter)
                    #grab only the x y z coords as floats
                    frame.append([float(i) for i in line.split()[2:5]])
                        
                coord_acc.append(frame)
        outfile.close()
    multiple = np.array(coord_acc)
    times = np.array(time_acc)
    return multiple, times


def output_plain(filename, frame, density=1.90985,
                 xbox=(-100,100), ybox=(-100,100), zbox=(-100,100)):
    """
    Accepts a frame and outputs a file consistent with the one required for 
    the start of a LAMMPS simulation.
    source: caspar_klug, 7/23/22
    author: Alex Yeh
    """

    pnum = frame.shape[0]
    header = f"LAMMPS data file for {pnum} particles in a fluid\n\n{pnum} atoms\n1 atom types\n"
    box = format_boxes(xbox, ybox, zbox)
    title = "\nAtoms\n\n"
    
    with open(filename, 'w') as outfile:
        outfile.write(header)
        outfile.write(box)
        outfile.write(title)
        for i, part in enumerate(frame):
            line = f"{i+1} 1 1.0 {density:0.5e} "
            coords = " ".join([f"{val:0.5f}" for val in part])
            outfile.write(line+coords+'\n')


