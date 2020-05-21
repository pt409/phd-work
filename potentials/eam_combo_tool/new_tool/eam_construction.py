#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:59:40 2020

@author: cdt1801
"""

import numpy as np
import pandas as pd

combo_eam = "combo_eam.set"

df = pd.read_table("eam_params")
df = df.set_index("element")
df = df.rename(columns={"lambda":"lambda0"})

types = ["Ni","Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]
    
# Pairwise potential
def phi_aa(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    return lambda r : A*np.exp(-alpha*(r/re-1))/(1+(r/re-chi)**20)-B*np.exp(-beta*(r/re-1))/(1+(r/re-lambda0)**20)

# Density function
def f(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    return lambda r : fe*np.exp(-beta1*r/re-1)/(1+(r/re-lambda1)**20)

# Embedding function        
def F(re,fe,rhoe,rhos,alpha,beta,A,B,chi,lambda0,Fn0,Fn1,Fn2,Fn3,F0,F1,F2,F3,eta,Fe,ielement,amass,F4,beta1,lambda1,rhol,rhoh):
    rhon = rhol*rhoe
    rhoo = rhoh*rhoe
    return lambda rho : Fn0+Fn1*(rho/rhon-1)+Fn2*(rho/rhon-1)**2+Fn3*(rho/rhon-1)**3 if rho<rhon else (F0+F1*(rho/rhoe-1)+F2*(rho/rhoe-1)**2+F3*(rho/rhoe-1)**3 if rho<rhoo else Fe*(1-eta*np.log(rho/rhos))*(rho/rhos)**eta)

# compute tabulated potentials
for atom_type in types:
    df.loc[atom_type].to_dict()

# Write top of file
header = ["\n","\n","\n"," ".join(["%.d" % len(types)]+types)+"\n"]
#header += [Nrho, drho, Nr, dr, cutoff]
with open(combo_eam,"w+") as open_eam_file:
    open_eam_file.writelines()
    