#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:19:41 2020

@author: cdt1801
"""
import sys
sys.path.append('../')

from simple_lammps_wrapper import Lammps, alloy_md_properties

import numpy as np

Lammps.command(20)

N = 20

elements = ["Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]
min_pc = [9.5,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
max_pc = [35.4,26.8,11.9,7.7,14.3,9.2,12.6,9.8,8.8]
min_sum = 20.0
max_sum = 63.0

n = len(elements)

min_pc = 0.01*np.array(min_pc)
max_pc = 0.01*np.array(max_pc)
min_sum *= 0.01
max_sum *= 0.01

"""compositions = [#[{"Ni":0.717,"Cr":0.113,"Al":0.077,"Ta":0.011,"Mo":0.082}],
                #[{"Ni":0.6198,"Cr":0.2254,"Co":0.0754,"Al":0.0403,"Ta":0.0002,"W":0.0299,"Ti":0.0015,"Mo":0.0074}],
                [{"Ni":0.5671,"Cr":0.2405,"Co":0.0971,"Al":0.0376,"Ta":0.0076,"W":0.0235,"Mo":0.0266}],
                [{"Ni":0.6728,"Cr":0.2322,"Al":0.0292,"Ta":0.0095,"W":0.0290,"Mo":0.0272}],
                [{"Ni":0.5248,"Cr":0.2558,"Co":0.1460,"Al":0.0139,"Ta":0.0080,"W":0.0254,"Mo":0.0262}]]"""

with open("output","a+") as output:
    output.write("Ni\t"+"\t".join(elements)+"\tsfe\tdensity\tlattice\tC11\tC12\tC44\n")

i = 0
while i < N :
    composition = []
    pc_sum = 0
    for j,el in enumerate(elements):
        amount = (max_pc[j]-min_pc[j])*np.random.random()+min_pc[j]
        composition += [amount]
        pc_sum += amount
    if pc_sum <= min_sum or pc_sum >= max_sum:
        pass
    else:
        composition = [1-pc_sum]+composition
        #with open("output","a+") as output:
            #output.write("\t".join(["%.5s" % x for x in composition])+"\t")
        print("\t".join(["%.5s" % x for x in composition])+"\t")
        composition = dict(zip(["Ni"]+elements,composition))
        print("alloy_%d" %i)
        #sfe,rho,a,c11,c12,c44 = alloy_md_properties(composition,"alloy_%d" %i,'sfe','density','lattice','C11','C12','C44')
        #with open("output","a+") as output:
            #output.write("%.5s\t%.5s\t%.6s\t%.5s\t%.5s\t%.5s\n" % (sfe,rho,a,c11,c12,c44))
        i += 1