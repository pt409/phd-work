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

dx = 0.1 # Angstroms
x = 0.0

Lammps.command(6,lammps_path="lammps")

initial_run = Lammps.setup('stacking_fault_1.in')

initial_run = Lammps.update(initial_run,"run",update_dict={"variable x_displace":"equal 0.0"})

displacement = []
E = []
while x < 3.52*np.sqrt(6)/2:
    current_run = Lammps.update(initial_run,str(x),update_dict={"variable x_displace":"equal "+str(x)})
    current_run.run()
    displacement += [x]
    x += dx
    with open(current_run.work_dir+"/"+current_run.log_file) as read_file:
        f = read_file.readlines()
        E += [float(f[-2].split()[-2])]
        
plt.plot(displacement,E)
plt.xlabel("Displacement (A)")
plt.ylabel("SFE energy (mJ/m^2)")