# CurvedSurfaceAnalyses
A package of code for analyzing the output of Monte Carlo/Brownian Dynamics simulations on curved surfaces.

Simulation engines: Hoomd-Blue, LAMMPS, custom monte carlo.

main_lib contains modules with analysis functions. Several example usage scripts sit on top.

Within main_lib:
       FileHandling.py reads in and writes out file types from the simulation engines
       GeometryHelpers.py contains several commonly used geometric methods
       UnitConversions.py converts quantities between SI units and simulation units (kT, 2a)
       IcosahedralShell.py creates Nx3 arrays of points on sphere in a mostly 6-coordinated, icosahedrally symmetric, lattice
       MSD.py computes mean-squared displacements, and several variants, for simulation trajectories
       Correlation.py mainly computes radial distribution functions, but can also spatially correlate other quantities
       OrderParameters.py contains several functions which characterize a snapshot of a simulation trajectory, coordination numbers, crsytallographic information, etc.
       ForceBalanceTheory.py uses an equation of state to balance osmotic pressures and predict an inhomogeneous density profile for an ensemble confined by an electric field.
       

Users can add this code to their python paths using conda-build.