#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:59:40 2020

@author: Pat Taylor (pt409)
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("../../../python-lammps")
from simple_lammps_wrapper import Lammps
from scipy.optimize import dual_annealing
from copy import deepcopy

# Read in the correct input files
# After calling the script these should be:
#    constr eam_params [list of element types]
#    fit eam_params fitting_scripts [list of element types]
run_type = None
run_option = sys.argv[1]
if "constr" in run_option:
    run_type = 0
    param_file = sys.argv[2]
    types = sys.argv[3:]
elif "fit" in run_option:
    run_type = 1
    param_file = sys.argv[2]
    fitting_scripts = sys.argv[3]
    types = sys.argv[4:]

Lammps.command(6,lammps_path="lammps")

df = pd.read_table(param_file)
df = df.set_index("element")
df = df.rename(columns={"lambda":"lambda0"})
#types=["Ni","Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]
#types=["Ni","Ru"]

# Track number of eam trials
trial_num = 0

# Pairwise potential
def phi(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,
        F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    return lambda r : A*np.exp(-alpha*(r/re-1))/(1+(r/re-chi)**20)-B*np.exp(-beta*(r/re-1))/(1+(r/re-lambda0)**20)

# Density function
def f(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,
      F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    return lambda r : fe*np.exp(-beta1*r/re-1)/(1+(r/re-lambda1)**20)

# Embedding function        
def F(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,
      F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh,s=1.0):
    rhoe *= s
    rhos *= s
    rhon = rhol*rhoe
    rhoo = rhoh*rhoe
    def conditional(rho):
        if rho<rhon:
            return Fn0+Fn1*(rho/rhon-1)+Fn2*(rho/rhon-1)**2+Fn3*(rho/rhon-1)**3
        elif rho<rhoo:
            return F0+F1*(rho/rhoe-1)+F2*(rho/rhoe-1)**2+F3*(rho/rhoe-1)**3
        else :
            return Fe*(1-eta*np.log(rho/rhos))*(rho/rhos)**eta
    return np.vectorize(conditional)

# Cross species pair potentials
def cross(types,tab_dens,tab_pots):
    for a,type_a in enumerate(types):
        for b,type_b in enumerate(types):
            if a<b:
                f_a = tab_dens[type_a]
                f_b = tab_dens[type_b]
                phi_aa = tab_pots[(type_a,type_a)]
                phi_bb = tab_pots[(type_b,type_b)]
                new_pair_pots = 0.5*(f_b/f_a*phi_aa+f_a/f_b*phi_bb)
                tab_pots[(type_a,type_b)] = new_pair_pots

# Write a .setfl tabulated eam file.
def write_setfl(name,types,Nrho,drho,Nr,dr,rc,atom_nums,atom_mass,equ_dists,tab_embs,tab_dens,tab_pots):
    # Write top of file
    # Format as specified for a .setfl file to use with eam/alloy https://lammps.sandia.gov/doc/pair_eam.html
    header = ["\n","\n","\n"] # Blank lines
    header += [" ".join(["{}".format(len(types))]+types)+"\n"] # line 4
    header += ["{} {:.15E} {} {:.15E} {:.15E}\n".format(Nrho,drho,Nr,dr,rc)] # line 5
    with open(name,"a+") as open_eam_file:
        open_eam_file.writelines(header)
        for atom_type in types:
            open_eam_file.write("{} {:.3f} {:.3f} fcc\n".format(atom_nums[atom_type],atom_mass[atom_type],equ_dists[atom_type]))
            np.savetxt(open_eam_file,tab_embs[atom_type].reshape((-1,5)))
            np.savetxt(open_eam_file,tab_dens[atom_type].reshape((-1,5)))
        for a,type_a in enumerate(types):
            for b,type_b in enumerate(types):
                if a<=b:
                    np.savetxt(open_eam_file,tab_pots[(type_a,type_b)].reshape((-1,5)))

# Write an eam .setfl w/o applying any gauge transformation.
def initial_eam(types,params=df,name="combo_eam.set",Nr=2000,Nrho=2000):
    rc = np.sqrt(5)*params.loc[types,"re"].max()
    rhoe_max = params.loc[types,"rhoe"].max()
    r,dr = np.linspace(0.0,rc,Nr,retstep=True)
    atom_nums = params.loc[types,"ielement"].to_dict()
    atom_mass = params.loc[types,"amass"].to_dict()
    equ_dists = (np.sqrt(2)*params.loc[types,"re"]).to_dict()
        
    # compute tabulated density and pairwise (same type i.e. a-a) potentials.
    tab_dens = {}
    tab_pots = {}
    f_max_as = np.zeros_like(types,dtype=np.float64)
    for a,atom_type in enumerate(types):
        densities = f(**params.loc[atom_type].to_dict())(r)
        pair_pots = phi(**params.loc[atom_type].to_dict())(r)
        f_max_a = densities.max()
        f_max_as[a] = f_max_a
        tab_dens[atom_type] = densities
        tab_pots[(atom_type,atom_type)] = pair_pots
    # Calculate largest density value to use for tabulated embedding fuctional.
    f_max = f_max_as.max()
    rho_max = max([100.0,f_max,2*rhoe_max])
    rho,drho = np.linspace(0.0,rho_max,Nrho,retstep=True)
    # compute tabulated embedding functional.
    tab_embs = {}
    for atom_type in types:
        embed_engs = F(**params.loc[atom_type].to_dict())(rho)  
        tab_embs[atom_type] = embed_engs
    # compute tabulated pairwise potentials.
    cross(types,tab_dens,tab_pots)
    
    # Write the .setfl file
    write_setfl(name,types,Nrho,drho,Nr,dr,rc,atom_nums,atom_mass,equ_dists,tab_embs,tab_dens,tab_pots)
    
    return tab_dens,tab_pots,tab_embs,f_max_as,dr,rc,atom_nums,atom_mass,equ_dists

# Write an eam /setfl after applying a gauge transformation
def transform_eam(types,k,s,tab_dens,tab_pots,tab_embs,
                  f_max_as,dr,rc,atom_nums,atom_mass,equ_dists,
                  params=df,name="trial_eam.set",Nr=2000,Nrho=2000):
    # Copy the dictionaries across
    tab_dens_t = deepcopy(tab_dens)
    tab_pots_t = deepcopy(tab_pots)
    tab_embs_t = deepcopy(tab_embs)
    # transform the density and pair potentials
    for a,atom_type in enumerate(types):
        tab_pots_t[(atom_type,atom_type)] -= 2*k[a]*tab_dens_t[atom_type]
        tab_dens_t[atom_type] *= s[a]
    # transform the cross-species pair potentials
    cross(types,tab_dens_t,tab_pots_t)
    # calculate rho_max
    f_max = (s*f_max_as).max()
    rhoe_max = (s*(params.loc[types,"rhoe"].to_numpy())).max()
    rho_max = max([100.0,f_max,2*rhoe_max])
    rho,drho = np.linspace(0.0,rho_max,Nrho,retstep=True)
    # calculate tabulated embedding function
    for atom_type,s_a,k_a in zip(types,s,k):
        embed_engs = F(**params.loc[atom_type].to_dict(),s=s_a)(rho) + k_a/s_a*rho  
        tab_embs_t[atom_type] = embed_engs
    # write the .setfl file
    write_setfl(name,types,Nrho,drho,Nr,dr,rc,atom_nums,atom_mass,equ_dists,tab_embs_t,tab_dens_t,tab_pots_t)

# Read in an input file listing lammps scripts and target properties.
def read_in_fitting(file_in):
    script_targets = {}
    script_insts = {}
    with open(file_in,"r") as f:
        for l in f:
            row = l.split()
            script_targets[row[0]] = np.array(row[1:],dtype=np.float64)
            script_insts[row[0]] = Lammps.setup(row[0])
    return script_targets,script_insts

# Objective function to use with a minimiser. 
def objective_function(x,types,
                       tab_dens,tab_pots,tab_embs,f_max_as,dr,rc,atom_nums,atom_mass,equ_dists,
                       lmp_scripts,lmp_insts,log_dir="logs"):
    global trial_num
    k,s = tuple(np.c_[[0.,1.],x.reshape(2,-1)]) # Add extra paramters for reference element.
    # give some output about the current iteration
    #for el,k_el,s_el in zip(types,k,s):
    print("Iteration # {}.\n".format(trial_num))
    print("\t".join(["k_{} = {:.5f}".format(el,k_el) for el,k_el in zip(types,k)])+"\n")
    print("\t".join(["s_{} = {:.5f}".format(el,s_el) for el,s_el in zip(types,s)])+"\n")
    # rm old potential and create new one with new k,s parameters.
    if os.path.exists("trial_eam.set"): os.remove("trial_eam.set")
    transform_eam(types,k,s,tab_dens,tab_pots,tab_embs,f_max_as,dr,rc,atom_nums,atom_mass,equ_dists)
    cost = 0
    for script in lmp_scripts:
        # Clean up previous runs.
        if os.path.exists(lmp_insts[script].log_loc()): 
            os.rename(lmp_insts[script].log_loc(),"{}/iter{}_{}".format(log_dir,trial_num,lmp_insts[script].log_file))
        # Run lammps
        lmp_insts[script].run()
        prop_list = ["property_{}".format(i+1) for i,_ in enumerate(lmp_scripts[script])]
        predictions = lmp_insts[script].read_log(prop_list,np_out=False,incl_step=False)
        # Loop throught predicted properties---check if lammps actually returned them---and work out cost.
        for j,pred in enumerate(predictions):
            actual = lmp_scripts[script][j]
            if pred: 
                print("Target value = {:.4f}\tpredicted value = {:.4f}\n".format(actual,pred[-1]))
                cost += ((actual-pred[-1])/actual)**2
    print("Total cost = {:.5f} for this iteration.\n".format(cost))
    trial_num += 1
    return cost

##############################################################################
# Run through any argv options provided to script.
if run_type == 0:
    eam_name = "".join(types)+".set"
    initial_eam(types,name=eam_name)
if run_type == 1:
    eam_name_0 = "".join(types)+".set"
    eam_name_f = "".join(types)+"-fitted.set"
    # Need a file to store lammps log files.
    if not os.path.exists("logs"):
        os.mkdir("logs")
    out_0 = initial_eam(types,name=eam_name_0)
    out_1 = read_in_fitting(fitting_scripts) 
    # Use simulated annealing to fit k,s.
    fit_result = dual_annealing(objective_function,np.array([[-2.0,1.0],[0.6,1.4]]),
                   maxiter=5000,
                   args=tuple([types])+out_0+out_1,
                   maxfun=1e5)
    k,s = tuple(np.c_[[0.,1.],(fit_result.x).reshape(2,-1)])
    transform_eam(types,k,s,*out_0,name=eam_name_f)