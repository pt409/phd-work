#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:18:35 2020

@author: cdt1801
"""

import numpy as np

elements = ["Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]
min_pc = [9.5,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
max_pc = [35.4,26.8,11.9,7.7,14.3,9.2,12.6,9.8,8.8]
min_sum = 20.0
max_sum = 63.0

N = 100 # total number of compositions to generate
n = len(elements)
m = 0

with open("alloys.out","a+") as output:
    output.write("\t".join(elements)+"\n")

while m < N:
    composition = []
    pc_sum = 0
    for i,el in enumerate(elements):
        amount = (max_pc[i]-min_pc[i])*np.random.random()+min_pc[i]
        composition += [amount]
        pc_sum += amount
    if pc_sum <= min_sum or pc_sum >= max_sum:
        pass
    else:
        composition = [100-pc_sum]+composition
        with open("alloys.out","a+") as output:
            output.write("\t".join(["%.4s" % x for x in composition])+"\n")
        m += 1