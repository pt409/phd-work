#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:21:45 2019

@author: cdt1801
"""
# This class allows for alloys to be set up with unqiue compositions, partitioned into phases, and assigned properties

import numpy as np
from mendeleev import element

class Material :
  
    def __init__(self,mol_pc=None) :
        self.composition = self.check_pc(mol_pc)
        self.mol_mass = sum([0.01*self.composition[el]*element(el).mass for el in self.composition])
        
        # Experimental data
        self.exp_density = np.array([])
        self.exp_cte = np.array([])
        self.exp_elastic = np.array([])
        self.exp_a = np.array([])
        
        # Simulated data (md)
        self.md_raw_dat = np.array([])
        self.md_density = np.array([]).reshape(2,0)
        self.md_cte = np.array([])
        self.md_elastic = np.array([]).reshape(2,0)
        self.md_a = np.array([]).reshape(2,0)
        self.md_C11 = np.array([])
        self.md_C12 = np.array([])
        self.md_C44 = np.array([])
        
    def atom_numbers(self,N) :
        num = {}
        tot = 0 # use to check that assigned atoms adds up to total N
        for el in self.composition :
            n = int(0.01*self.composition[el]*N)
            num[el] = n
            tot += n
        # assign any remainder to element with largest number of corresponding atoms.
        bal_key = max(self.composition,key=self.composition.get)
        num[bal_key] += N - tot
        
        return num

    def calc_lattice_const(self) :
        # Anticipate densities to be an array of temperatures and densities
        a = np.array(self.md_density)
        a[1] = (a[1]*6.02214076E-1/(4*self.mol_mass))**(-1/3)
        self.md_a = a
    
    @classmethod
    def weight_pc(cls,wt_pc) :
        wt_pc_2 = cls.check_pc(wt_pc)
        mol_pc = {}
        mol_tot = 0
        for m in wt_pc_2 :
            mol_tot += wt_pc_2[m]/element(m).mass
    
        for m in wt_pc_2 :
            mol_pc[m] = round(100*(wt_pc_2[m]/element(m).mass)/mol_tot,4)
        return cls(mol_pc)
    
    @staticmethod
    def check_pc(pc_in) :
        tot = 0
        pc_out = {}
        balance = None
        for m in pc_in:
            if pc_in[m] == 0 :
                continue
            elif isinstance(pc_in[m],float) or isinstance(pc_in[m],int) :
                pc_out[m] = pc_in[m]
                tot+=pc_in[m]
            else: 
                balance = m
        
        if balance != None :
            pc_out[balance] = 100-tot
        else :
            for m in pc_out:
                pc_out[m]=100*pc_in[m]/tot
            
        return pc_out
    
class Alloy(Material) :
    def __init__(self,mol_pc=None) :
        self.gamma = None
        self.gammap = None
        self.struc = 'fcc' # Structure is assumed to be fcc unless otherwise specified. 'mix' and 'L12' also allowed.
        super().__init__(mol_pc)   
    
    # A crude model that partitions the alloy into gamma and gammap phases
    def partition(self) :
        bulk_comp = self.composition
        g_comp = {}
        g_tot = 0
        gp_comp = {}
        gp_tot = 0
        for el in bulk_comp :
            if el in ('Al','Ti','Ta') :
                n = 1*bulk_comp[el]
                gp_comp[el] = n 
                gp_tot += n
            elif el != 'Ni':
                n = 1*bulk_comp[el]
                g_comp[el] = n 
                g_tot += n
        # gamma' phase can have sublattices
        sublat1 = {**gp_comp}
        sublat2 = {'Ni':3*gp_tot}
        # Final compositions for each phase
        gp_comp['Ni'] = 3*gp_tot
        gp_tot *= 4
        g_comp['Ni'] = bulk_comp['Ni'] - gp_comp['Ni']
        g_tot += g_comp['Ni']
        
        # correct to be proper percentages
        for tot,phase in zip([g_tot,gp_tot],[g_comp, gp_comp]) :
            for el in phase:
                if tot != 0 :
                    phase[el] *= 100/tot
        
        if g_tot != 0 :
            self.gamma = Phase(g_comp,g_tot)
        if gp_tot != 0 :
            self.gammap = Phase(gp_comp,gp_tot)
            self.gammap.sublat = Material(sublat1), Material(sublat2)
            self.struc = 'L12'
        if g_tot !=0 and gp_tot !=0 :
            self.struc = 'mix'
     
    # Related to the partitioning model above, use it to compute the bulk density of the alloy
    def calc_bulk_density(self) :
        if self.struc == 'mix' :
            temps = sorted(list(set(self.gamma.md_density[0])|set(self.gammap.md_density[0])))
            for T in temps:
                if T in self.gamma.md_density[0] and T in self.gammap.md_density[0]:
                    i = np.where(self.gamma.md_density[0] == T)[0][0]
                    j = np.where(self.gammap.md_density[0] == T)[0][0]
                    rho = 0.01*(self.gamma.mol_frac*self.gamma.md_density[1][i] + self.gammap.mol_frac*self.gammap.md_density[1][j])
                    self.md_density = np.c_[self.md_density,[[T],[rho]]]
        elif self.struc == 'L12' :
            self.md_density = 1*self.gammap.md_density
        elif self.struc == 'fcc' and not self.gamma :
            self.md_density = 1*self.gamma.md_density
            
class Phase(Material) :
    def __init__(self,mol_pc=None,mol_frac=0) :
        self.mol_frac = mol_frac # Molar fraction of bulk alloy that this phase comprises
        self.sublat = None
        super().__init__(mol_pc) 