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
import matplotlib.pyplot as plt

start_time = time.time()

################################ PARAMETERS ###################################

dx = -0.2 # Angstroms
x = 0.0
A = (3.52*np.sqrt(6)/2*10)*(3.52*np.sqrt(2)/2*10)

Lammps.command(6,lammps_path="lammps")

initial_run = Lammps.setup('stacking_fault_npt.in')

initial_run = Lammps.update(initial_run,"npt_run",update_dict={"variable x_displace":"equal 0.0"})

displacement = []
E = []
while abs(x) < 3.52*np.sqrt(6)/2:
    current_run = Lammps.update(initial_run,str(x),update_dict={"variable x_displace":"equal "+str(x)})
    current_run.run()
    displacement += [x]
    x += dx
    #with open(current_run.work_dir+"/"+current_run.log_file) as read_file:
    #    f = read_file.readlines()
    #    E += [float(f[-2].split()[-2])]
    output = current_run.read_log(["TotEng","Temp","Press"]) 
    E += [np.mean(output[1][100000:])*1.60217657e4/A] # conversion factor from eV/A^2 to mJ/m^2
    
plt.plot(displacement,E,'bo-')
plt.xlabel("Displacement (A)")
plt.ylabel("SFE energy (mJ/m^2)")