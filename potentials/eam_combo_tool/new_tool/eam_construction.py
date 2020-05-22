#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:59:40 2020

@author: cdt1801
"""

import numpy as np
import pandas as pd
import time

combo_eam = "combo_eam.set"

df = pd.read_table("eam_params")
df = df.set_index("element")
df = df.rename(columns={"lambda":"lambda0"})

types = ["Ni","Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]

t0 = time.time()

Nr = 2000
Nrho = 2000
rc = np.sqrt(5/4)*df.loc[types,"re"].max()
rhoe_max = df.loc[types,"rhoe"].max()
r,dr = np.linspace(0.0,rc,Nr,retstep=True)
atom_nums = df.loc[types,"ielement"].to_dict()
atom_mass = df.loc[types,"amass"].to_dict()
equ_dists = (np.sqrt(2)*df.loc[types,"re"]).to_dict()


# Pairwise potential
def phi(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    return lambda r : A*np.exp(-alpha*(r/re-1))/(1+(r/re-chi)**20)-B*np.exp(-beta*(r/re-1))/(1+(r/re-lambda0)**20)

# Density function
def f(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    return lambda r : fe*np.exp(-beta1*r/re-1)/(1+(r/re-lambda1)**20)

# Embedding function        
def F(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    rhon = rhol*rhoe
    rhoo = rhoh*rhoe
    #return lambda rho : Fn0+Fn1*(rho/rhon-1)+Fn2*(rho/rhon-1)**2+Fn3*(rho/rhon-1)**3 if rho<rhon else (F0+F1*(rho/rhoe-1)+F2*(rho/rhoe-1)**2+F3*(rho/rhoe-1)**3 if rho<rhoo else Fe*(1-eta*np.log(rho/rhos))*(rho/rhos)**eta)
    #return lambda rho : np.where(rho<rhon,Fn0+Fn1*(rho/rhon-1)+Fn2*(rho/rhon-1)**2+Fn3*(rho/rhon-1)**3,
    #                             np.where(rho<rhoo,F0+F1*(rho/rhoe-1)+F2*(rho/rhoe-1)**2+F3*(rho/rhoe-1)**3,
    #                                      Fe*(1-eta*np.log(rho/rhos))*(rho/rhos)**eta))
    def conditional(rho):
        if rho<rhon:
            return Fn0+Fn1*(rho/rhon-1)+Fn2*(rho/rhon-1)**2+Fn3*(rho/rhon-1)**3
        elif rho<rhoo:
            return F0+F1*(rho/rhoe-1)+F2*(rho/rhoe-1)**2+F3*(rho/rhoe-1)**3
        else :
            return Fe*(1-eta*np.log(rho/rhos))*(rho/rhos)**eta
    return np.vectorize(conditional)

# compute tabulated density and pairwise (same type i.e. a-a) potentials.
tab_dens = {}
tab_pots = {}
f_max = 0.0
for atom_type in types:
    densities = f(**df.loc[atom_type].to_dict())(r)
    pair_pots = phi(**df.loc[atom_type].to_dict())(r)
    f_max_i = densities.max()
    f_max = f_max_i if f_max_i > f_max else f_max
    tab_dens[atom_type] = densities
    tab_pots[(atom_type,atom_type)] = pair_pots
# Calculate largest density value to use for tabulated embedding fuctional.
rho_max = max([100.0,f_max,2*rhoe_max])
rho,drho = np.linspace(0.0,rho_max,Nrho,retstep=True)
# compute tabulated embedding functional.
tab_embs = {}
for atom_type in types:
    embed_engs = F(**df.loc[atom_type].to_dict())(rho)  
    tab_embs[atom_type] = embed_engs
# compute tabulated pairwise potentials.
for a,type_a in enumerate(types):
    for b,type_b in enumerate(types):
        if a<b:
            f_a = tab_dens[type_a]
            f_b = tab_dens[type_b]
            phi_aa = tab_pots[(type_a,type_a)]
            phi_bb = tab_pots[(type_b,type_b)]
            new_pair_pots = 0.5*(f_b/f_a*phi_aa+f_a/f_b*phi_bb)
            tab_pots[(type_a,type_b)] = new_pair_pots

# Write top of file
# Format as specified for a .setfl file to use with eam/alloy https://lammps.sandia.gov/doc/pair_eam.html
header = ["\n","\n","\n"] # Blank lines
header += [" ".join(["{}".format(len(types))]+types)+"\n"] # line 4
header += ["{} {:.15E} {} {:.15E} {:.15E}\n".format(Nrho,drho,Nr,dr,rc)] # line 5
with open(combo_eam,"a+") as open_eam_file:
    open_eam_file.writelines(header)
    for atom_type in types:
        open_eam_file.write("{} {:.3f} {:.3f} fcc\n".format(atom_nums[atom_type],atom_mass[atom_type],equ_dists[atom_type]))
        np.savetxt(open_eam_file,tab_embs[atom_type].reshape((-1,5)))
        np.savetxt(open_eam_file,tab_dens[atom_type].reshape((-1,5)))
    for a,type_a in enumerate(types):
        for b,type_b in enumerate(types):
            if a<=b:
                np.savetxt(open_eam_file,tab_pots[(type_a,type_b)].reshape((-1,5)))
        
t1 = time.time()
print("Total time to write potential = {:.3f}s".format(t1-t0))