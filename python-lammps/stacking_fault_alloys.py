#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:24:58 2019
@author: cdt1801
"""
# Script to run a given set of simulations for given input alloys
import numpy as np
import subprocess as sp

import sys
sys.path.append('../')
import os

from simple_lammps_wrapper import Lammps
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start_time = time.time()

################################ PARAMETERS ###################################

steps = 40
x = 0.0
alloy_comp = {"Ni":0.5219,"Cr":0.271,"Co":0.1181,"Al":0.0369,"Ta":0.0042,"W":0.0153,"Ti":0.0036,"Mo":0.029} # AM3 gamma matrix
alloy_comp = {"Ni":0.5493,"Cr":0.2558,"Co":0.0921,"Al":0.0284,"Ta":0.0029,"W":0.0423,"Ti":0.0025,"Mo":0.0267} # MC2 gamma matrix
alloy_comp = {"Ni":0.608,"Cr":0.236,"Co":0.082,"Al":0.033,"Ta":0.014,"W":0.02,"Ti":0.007} # Alloy 454 gamma matrix
alloy_comp = {"Ni":0.596,"Cr":0.255,"Co":0.086,"Al":0.031,"Ta":0.001,"W":0.025,"Ti":0.006} # CMSX-2 gamma matrix
alloy_comp = {"Ni":0.59,"Cr":0.155,"Co":0.136,"Al":0.039,"Ta":0.006,"W":0.074} # TMS-1 gamma matrix
alloy_comp = {"Ni":0.558,"Cr":0.265,"Co":0.096,"Al":0.034,"Ta":0.007,"W":0.035,"Ti":0.005} # PWA 1480 gamma matrix
alloy_comp = {"Ni":0.54,"Cr":0.261,"Co":0.111,"Al":0.03,"Ta":0.007,"W":0.023,"Ti":0.004,"Mo":0.025} # AM1 gamma matrix
alloy_comp = {"Ni":0.5491,"Cr":0.3129,"Co":0.0762,"Al":0.0321,"Ta":0.0023,"W":0.0145,"Mo":0.0091} # STAL-15 gamma matrix
alloy_comp = {"Ni":1.00}

alloy_elements = " ".join(sorted(alloy_comp.keys()))

Lammps.command(6,lammps_path="lammps")

base_dir = "alloy_sfe_script"

sp.call(["mkdir","-p",base_dir])
sfe_setup = Lammps.default_setup("sfe_setup",loc=base_dir)
sfe_setup.run() # This produces a datafile elemental.data
sfe_min = Lammps.default_setup("sfe_min",loc=base_dir)
sfe_min.data_file = "elemental.data"
sfe_min.alloyify(alloy_comp.copy(),alloy_comp.copy())
sfe_min.update(update_dict={"read_data":"alloyified_elemental.data"},replace_dict={"Ni Ni":alloy_elements,"Ni Al Ni Al":alloy_elements})
sfe_min.run()
# Find the maximum displacement that will return cell to original equilibrium
with open(sfe_min.log_loc()) as read_file:
    f = read_file.readlines()
    x_tot = float(f[-17].split()[-2])
a = x_tot*2/np.sqrt(6) # Lattice parameter
dx = -x_tot/2/(steps-1)
# Step refers to shift in upper part of supercell to calculate intrinsic stacking fault
sfe_step = Lammps.default_setup("sfe_step",loc=base_dir)
sfe_step.update(update_dict={"variable x_displace":"equal 0.0","read_data":"alloyified_elemental.data","variable latparam1 equal":str(a)},replace_dict={"Ni Ni":alloy_elements,"Ni Al Ni Al":alloy_elements})
sfe_step.run()
with open(sfe_step.log_loc()) as read_file:
    f = read_file.readlines()
    E = float(f[-2].split()[-2])

displacement = []
E = []
for step in range(steps):
    current_run = Lammps.based_on_setup(sfe_step,"step_"+str(step)+".in",update_dict={"variable x_displace":"equal "+str(x)})
    current_run.run()
    displacement += [x/a]
    x += dx
    with open(current_run.log_loc()) as read_file:
        f = read_file.readlines()
        E += [float(f[-2].split()[-2])]
        
E = np.array(E)
E -= E[0]

# Find extrema, and their nature
#first = [(E[(i+1)%steps]-E[i-1])/(2*dx) for i,_ in enumerate(E)]
#second = [(E[i-1]-2*E[i]+E[(i+1)%steps])/dx**2 for i,_ in enumerate(E)]
        
# put a line through sf, usf, utf
plt.plot([-1/2/np.sqrt(6),-1/2/np.sqrt(6)],[0,E.max()],'--k')
plt.plot([-1/np.sqrt(6),-1/np.sqrt(6)],[0,E.max()],'--k')
plt.plot([-3*1/2/np.sqrt(6),-3*1/2/np.sqrt(6)],[0,E.max()],'--k')
# plot actual SFE now
plt.plot(displacement,E,'c')
plt.xlabel("Displacement/lattice parameter")
plt.ylabel("SFE energy (mJ/m^2)")
plt.savefig("alloy_test.png",dpi=400)
plt.close()