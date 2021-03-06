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

Lammps.command(6,lammps_path="lammps")

# Find relaxed cell
init_1 = Lammps.setup('sfe_1.in')
init_1 = Lammps.update(init_1,"sfe_script")
init_1.run()
init_2 = Lammps.setup('sfe_2.in')
init_2 = Lammps.update(init_2,"sfe_script/init",new_data_file="sfe_script/elemental.data")
init_2.run()

with open(init_2.log_loc()) as read_file:
        f = read_file.readlines()
        x_tot = float(f[-17].split()[-2]) # The maximum displacement that will return cell to original equilibrium

# This is just to setup correct file structure
setup_run = Lammps.setup('sfe_3_restart.in')
setup_run = Lammps.update(setup_run,"sfe_script/run",update_dict={"variable x_displace":"equal 0.0"})

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
plt.savefig("test.png",dpi=400)
plt.close()