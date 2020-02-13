#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:19:41 2020

@author: cdt1801
"""
import sys
sys.path.append('../')

from simple_lammps_wrapper import Lammps, alloy_md_properties

Lammps.command(20)


compositions = [[{"Ni":0.717,"Cr":0.113,"Al":0.077,"Ta":0.011,"Mo":0.082}],
                [{"Ni":0.6198,"Cr":0.2254,"Co":0.0754,"Al":0.0403,"Ta":0.0002,"W":0.0299,"Ti":0.0015,"Mo":0.0074}],
                [{"Ni":0.5671,"Cr":24.05,"Co":9.71,"Al":3.76,"Ta":0.76,"W":2.35,"Mo":2.66}],
                [{"Ni":0.6728,"Cr":0.2322,"Al":0.0292,"Ta":0.0095,"W":0.0290,"Mo":0.0272}],
                [{"Ni":0.5248,"Cr":0.2558,"Co":0.1460,"Al":0.0139,"Ta":0.0080,"W":0.0254,"Mo":0.0262}]]

for i,composition in enumerate(compositions):
    sfe,rho,a,c11,c12,c44 = alloy_md_properties(composition,"alloy_%d" %i,'sfe','density','lattice','C11','C12','C44')
    with open("output","a+") as output:
            output.write("%.4s\t%.4s\t%.5s\t%.5s\t%.5s\t%.5s\n" % (sfe,rho,a,c11,c12,c44))