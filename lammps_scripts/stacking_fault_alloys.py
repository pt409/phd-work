#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:24:58 2019
@author: cdt1801
"""
# Script to run a given set of simulations for given input alloys
import numpy as np
import subprocess as sp
import os

import sys
sys.path.append('../')

from simple_lammps_wrapper import Lammps
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start_time = time.time()

################################ PARAMETERS ###################################

steps = 30
x = 0.0
alloy_comp = {"Ni":0.5219,"Cr":0.271,"Co":0.1181,"Al":0.0369,"Ta":0.0042,"W":0.0153,"Ti":0.0036,"Mo":0.029} # AM3 gamma matrix
#alloy_comp = {"Ni":0.5493,"Cr":0.2558,"Co":0.0921,"Al":0.0284,"Ta":0.0029,"W":0.0423,"Ti":0.0025,"Mo":0.0267} # MC2 gamma matrix

alloy_elements = " ".join(sorted(alloy_comp.keys()))

Lammps.command(6,lammps_path="lammps")

base_dir = "alloy_sfe_script"

# Find relaxed cell
init_1 = Lammps.setup('init_1.in')
init_1 = Lammps.update(init_1,base_dir)
init_1.run()
init_2 = Lammps.setup('init_2.in')
init_2 = Lammps.update(init_2,base_dir+"/init",update_dict={"read_data":"alloyified_elemental.data","pair_coeff":"* * NiAlCoCrMoTiWTa.set "+alloy_elements},new_data_file=base_dir+"/elemental.data")
init_2.alloyify(alloy_comp.copy(),alloy_comp.copy())
init_2.run()

with open(init_2.log_loc()) as read_file:
        f = read_file.readlines()
        x_tot = float(f[-17].split()[-2]) # The maximum displacement that will return cell to original equilibrium

# This is just to setup correct file structure
setup_run = Lammps.setup('stacking_fault_min_restart.in')
setup_run = Lammps.update(setup_run,base_dir+"/run",update_dict={"variable x_displace":"equal 0.0","pair_coeff":"* * NiAlCoCrMoTiWTa.set "+alloy_elements})

a = x_tot*2/np.sqrt(6)
dx = -x_tot/2/(steps-1)

displacement = []
E = []
for step in range(steps):
    current_run = Lammps.update(setup_run,"step_"+str(step),update_dict={"variable x_displace":"equal "+str(x)})
    current_run.run()
    displacement += [x]
    x += dx
    with open(current_run.log_loc()) as read_file:
        f = read_file.readlines()
        E += [float(f[-2].split()[-2])]
        
E = np.array(E)
E -= E[0]

# Find extrema, and their nature
first = [(E[(i+1)%steps]-E[i-1])/(2*dx) for i,_ in enumerate(E)]
second = [(E[i-1]-2*E[i]+E[(i+1)%steps])/dx**2 for i,_ in enumerate(E)]
        
# put a line through sf, usf, utf
plt.plot([-a/2/np.sqrt(6),-a/2/np.sqrt(6)],[0,E.max()],'r')
plt.plot([-a/np.sqrt(6),-a/np.sqrt(6)],[0,E.max()],'r')
plt.plot([-3*a/2/np.sqrt(6),-3*a/2/np.sqrt(6)],[0,E.max()],'r')
# plot actual SFE now
plt.plot(displacement,E,'bo-')
plt.xlabel("Displacement (A)")
plt.ylabel("SFE energy (mJ/m^2)")
plt.savefig("alloy_test_AM3.png",dpi=400)
plt.close()