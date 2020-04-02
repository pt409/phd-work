#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:27:20 2020

@author: cdt1801
"""

import numpy as np
import pandas as pd
from mendeleev import element
import sys

input_database = sys.argv[1]
output_database = sys.argv[2]

df = pd.read_excel(input_database,engine="odf",header=[0,1,2])
els = df["Composition","wt. %"].columns.values
masses = np.array([element(el.strip()).mass for el in els])

# Convert from wt. % to at. %
def convert_2_at(wt_comp,masses):
    at_comp = wt_comp/masses
    at_comp *= 100/at_comp.sum()
    return at_comp
# Convert from at. % to wt. %
def convert_2_wt(at_comp,masses):
    wt_comp = at_comp*masses
    wt_comp *= 100/wt_comp.sum()
    return wt_comp

# Check if a subsection of a row is a valid input for processing purposes.
def check_if_valid(input_):
    if np.prod("-" == input_):
        return input_, 0 # Invalid input, all entries are -
    else :
        input_ = np.where(input_ == "-",0,input_).astype(np.float64)
        if np.isnan(input_.sum()):
            return input_, 1 # Invalid input, (at least one) empty column
        else :
            return input_, 2 # Valid input
        
# Calculate the wt. % chemical comp. from the at. % and vice versa.
def calc_wt_at_vv(wt_comp,at_comp,masses):
    return_code = 0 # Could not calculate complete compositions
    wt_comp, wt_return_code = check_if_valid(wt_comp)
    at_comp, at_return_code = check_if_valid(at_comp)
    if wt_return_code == 1 and at_return_code == 2 :
        wt_comp = convert_2_wt(at_comp,masses)
        return_code = 1 # Could calculate complete compositions (wt & at %)
    elif wt_return_code == 2 and at_return_code == 1 :
        at_comp = convert_2_at(wt_comp,masses)
        return_code = 1
    return wt_comp, at_comp, return_code

# Calculate the fraction of gamma' precipitate from the nominal, matrix and precipitate chemical compositions.
def calc_prc_frac(nom,mtx,prc):
    zero_vals = np.concatenate((np.nonzero(nom==0)[0],np.nonzero(mtx==0)[0],np.nonzero(prc==0)[0]))
    if zero_vals.size > 0:
        nom = np.delete(nom,zero_vals)
        mtx = np.delete(mtx,zero_vals)
        prc = np.delete(prc,zero_vals)
    return 100*np.mean((nom-mtx)/(prc-mtx))

# Process compositions (nominal, matrix, and precipitate as well as precipitate fractions.)
def process_composition(index,row,return_codes=False):
    # Convert any wt. % compositions to at. % and vice versa
    nom_wt_comp = row["Composition","wt. %"].values
    nom_at_comp = row["Composition","at. %"].values
    nom_wt_comp, nom_at_comp, nom_rtn_code = calc_wt_at_vv(nom_wt_comp,nom_at_comp,masses)
    mtx_wt_comp = row["γ composition","wt. %"].values
    mtx_at_comp = row["γ composition","at. %"].values
    mtx_wt_comp, mtx_at_comp, mtx_rtn_code = calc_wt_at_vv(mtx_wt_comp,mtx_at_comp,masses)
    prc_wt_comp = row["γ’ composition","wt. %"].values
    prc_at_comp = row["γ’ composition","at. %"].values
    prc_wt_comp, prc_at_comp, prc_rtn_code = calc_wt_at_vv(prc_wt_comp,prc_at_comp,masses)
    # Get the wt, at, vol percentages of the precipitates
    wt_frac,wt_frac_rtn_code=check_if_valid(row["γ’ fraction","wt. %","Unnamed: 41_level_2"])
    at_frac,at_frac_rtn_code=check_if_valid(row["γ’ fraction","at. %","Unnamed: 42_level_2"])
    vl_frac,vl_frac_rtn_code=check_if_valid(row["γ’ fraction","vol. %","Unnamed: 43_level_2"])
    # To a good approximation the at. % and vol. % are the same
    at_frac = 1*vl_frac if vl_frac_rtn_code==2 and at_frac_rtn_code!=2 else at_frac    
    at_frac_rtn_code = min(at_frac_rtn_code*vl_frac_rtn_code,2)
    if nom_rtn_code:
        # Case 2 and 3: have composition of the matrix phase and precipitate fracion but not the precipitate comp.
        if mtx_rtn_code and (not prc_rtn_code):
            if wt_frac_rtn_code==2: # Case 2: have wt frac
                prc_wt_comp = (nom_wt_comp-(1-0.01*wt_frac)*mtx_wt_comp)/(0.01*wt_frac)
            if at_frac_rtn_code==2: # Case 3: have at frac
                prc_at_comp = (nom_at_comp-(1-0.01*at_frac)*mtx_at_comp)/(0.01*at_frac)
            if (at_frac_rtn_code==2) ^ (wt_frac_rtn_code==2): # Compute at from wt comp and vv
                prc_wt_comp, prc_at_comp, prc_rtn_code = calc_wt_at_vv(prc_wt_comp,prc_at_comp,masses)
        # Case 4 and 5: have composition of the precipitate phase but not of the matrix (reverse of 2 and 3)
        elif (not mtx_rtn_code) and prc_rtn_code:
            if wt_frac_rtn_code==2: # Case 4: have wt frac
                mtx_wt_comp = (nom_wt_comp-0.01*wt_frac*prc_wt_comp)/(1-0.01*wt_frac)
            if at_frac_rtn_code==2: # Case 5: have at frac
                mtx_at_comp = (nom_at_comp-0.01*at_frac*prc_at_comp)/(1-0.01*at_frac)
            if (at_frac_rtn_code==2) ^ (wt_frac_rtn_code==2): # Compute at from wt comp and vv
                mtx_wt_comp, mtx_at_comp, mtx_rtn_code = calc_wt_at_vv(mtx_wt_comp,mtx_at_comp,masses)
        # Case 1: have compositions of each phase but not the precipitate fraction
        # Do this case at the end as it also accounts for cases above where at/wt frac need to be computed too.
        if mtx_rtn_code and prc_rtn_code:
            if wt_frac_rtn_code==1:
                wt_frac = calc_prc_frac(nom_wt_comp,mtx_wt_comp,prc_wt_comp)
                wt_frac_rtn_code = 2
            if at_frac_rtn_code==1:
                at_frac = calc_prc_frac(nom_at_comp,mtx_at_comp,prc_at_comp)
                at_frac_rtn_code = 2
    if not return_codes:
        return nom_wt_comp,nom_at_comp,mtx_wt_comp,mtx_at_comp,prc_wt_comp,prc_at_comp,wt_frac,at_frac
    else:
        return nom_wt_comp,nom_at_comp,nom_rtn_code,mtx_wt_comp,mtx_at_comp,mtx_rtn_code,prc_wt_comp,prc_at_comp,prc_rtn_code,wt_frac,wt_frac_rtn_code,at_frac,at_frac_rtn_code

# Compute Larson-Miller parameter
def lmp(temp_degC,time_hrs,lmp_param=20,factor=1e-3):
    return factor*(temp_degC+273.15)*(lmp_param+np.log10(time_hrs))

# Process creep data
def process_creep(index,row,stress_bins=None):  
    entry_data = pd.DataFrame(row["Creep life"].values.reshape(9,10)).sort_values(by=[1]).to_numpy()
    for j,entry_col in enumerate(entry_data):
        stress,temp,lmp_1pc,lmp_rupture,t_1pc,t_rupture = tuple(entry_col[:6])
        # Check whether stress entry is non-zero
        if not np.isnan(stress):
            # If temp entry is nonzero look for entries for non-LMP data (time data) and convert to LMP
            if not np.isnan(temp):
                if (not np.isnan(t_1pc)) and np.isnan(lmp_1pc): # 1% creep time
                    lmp_1pc = lmp(temp,t_1pc)
                if (not np.isnan(t_rupture)) and np.isnan(lmp_rupture): # Rupture time
                    lmp_rupture = lmp(temp,t_rupture)
        entry_data[j][2] = lmp_1pc
        entry_data[j][3] = lmp_rupture
    return entry_data.reshape(90)
    
# Will make a copy of the dataframe to process
df_proc = df.copy()

# Process the entire database and write it to the copied database.
for index, row in df.iterrows():
    nom_wt_comp,nom_at_comp,mtx_wt_comp,mtx_at_comp,prc_wt_comp,prc_at_comp,wt_frac,at_frac = process_composition(index,row)
    # Write these values to the copied database
    df_proc.loc[index,("Composition","wt. %")] = nom_wt_comp
    df_proc.loc[index,("Composition","at. %")] = nom_at_comp
    df_proc.loc[index,("γ composition","wt. %")] = mtx_wt_comp
    df_proc.loc[index,("γ composition","at. %")] = mtx_at_comp
    df_proc.loc[index,("γ’ composition","wt. %")] = prc_wt_comp
    df_proc.loc[index,("γ’ composition","at. %")] = prc_at_comp
    df_proc.loc[index,("γ’ fraction","wt. %","Unnamed: 41_level_2")] = wt_frac    
    df_proc.loc[index,("γ’ fraction","at. %","Unnamed: 42_level_2")] = at_frac
    df_proc.loc[index,("γ’ fraction","vol. %","Unnamed: 43_level_2")] = at_frac
    # Do the same thing for creep data
    new_creep_data = process_creep(index,row)
    df_proc.loc[index,"Creep life"] = process_creep(index,row)
    
# Write out the processed database to a csv file.
df_proc.to_csv(output_database)