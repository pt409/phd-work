#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:27:20 2020

@author: cdt1801
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mendeleev import element
import sys

#input_database = "../big_alloy_database.ods"
input_database = sys.argv[1]
output_database = sys.argv[2]
if "creep" in sys.argv[2:]:
    do_creep_processing = True
else: do_creep_processing = False
#do_creep_processing = False

df = pd.read_excel(input_database,engine="odf",header=[0,1,2])
els = df["Composition","wt. %"].columns.values
masses = np.array([element(el.strip()).mass for el in els])

#%% Processing functions.

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
def calc_prc_frac(nom,mtx,prc,error=0.01):
    zero_vals = np.concatenate((np.nonzero(nom<error)[0],np.nonzero(mtx<error)[0],np.nonzero(prc<error)[0]))
    if zero_vals.size > 0:
        nom = np.delete(nom,zero_vals)
        mtx = np.delete(mtx,zero_vals)
        prc = np.delete(prc,zero_vals)
    # Use method from Segersall et al 2015 of fitting a linear regression model.
    line = LinearRegression(fit_intercept=False)
    line.fit((prc-mtx).reshape(-1,1),(nom-mtx).reshape(-1,1))
    f = 100.*line.coef_[0,0]
    return f

def calc_part_coeff(mtx,prc,x_tol):
    K = np.full_like(mtx,np.nan)
    for i,(x_mtx,x_prc) in enumerate(zip(mtx,prc)):
        if abs(x_mtx) < x_tol and abs(x_prc) < x_tol:
            continue
        elif abs(x_mtx) < x_tol and abs(x_prc) >= x_tol:
            x_mtx += x_tol
        elif abs(x_mtx) >= x_tol and abs(x_prc) < x_tol:
            x_prc += x_tol
        K[i] = x_mtx/x_prc
    return K          
        
def read_in_part_coeff(input_):
    return np.array([np.nan if i == "-" else i for i in input_]).astype(np.float64)

def inf_2_large(input_,large_num):
    return np.array([large_num if np.isinf(_) else _ for _ in input_])

def replace_if_new_value(input_old,input_new):
    return np.array([old if np.isfinite(old) else new for old,new in zip(input_old,input_new)])

# Process compositions (nominal, matrix, and precipitate as well as precipitate fractions.)
def process_composition(index,row,return_codes=False,inf_value=1e3,x_tol=0.01,correct_if_neg=False):
    # Convert any wt. % compositions to at. % and vice versa
    nom_wt_comp = row[("Composition","wt. %")].values
    nom_at_comp = row[("Composition","at. %")].values
    nom_wt_comp, nom_at_comp, nom_rtn_code = calc_wt_at_vv(nom_wt_comp,nom_at_comp,masses)
    mtx_wt_comp = row[("γ composition","wt. %")].values
    mtx_at_comp = row[("γ composition","at. %")].values
    mtx_wt_comp, mtx_at_comp, mtx_rtn_code = calc_wt_at_vv(mtx_wt_comp,mtx_at_comp,masses)
    prc_wt_comp = row[("γ’ composition","wt. %")].values
    prc_at_comp = row[("γ’ composition","at. %")].values
    prc_wt_comp, prc_at_comp, prc_rtn_code = calc_wt_at_vv(prc_wt_comp,prc_at_comp,masses)
    # Get the partitioning coefficients
    K_part_wt = read_in_part_coeff(row[("γ/γ’ partitioning ratio","wt. %")].values)
    K_part_at = read_in_part_coeff(row[("γ/γ’ partitioning ratio","at. %")].values)
    # New things
    K2_part_wt = np.full(len(els),np.nan)
    K2_part_at = np.full(len(els),np.nan)
    lost_at    = np.full(len(els),np.nan)
    # Get the wt, at, vol percentages of the precipitates
    wt_frac,wt_frac_rtn_code=check_if_valid(row[("γ’ fraction","wt. %")])
    at_frac,at_frac_rtn_code=check_if_valid(row[("γ’ fraction","at. %")])
    vl_frac,vl_frac_rtn_code=check_if_valid(row[("γ’ fraction","vol. %")])
    # To a good approximation the at. % and vol. % are the same
    at_frac = 1*vl_frac if vl_frac_rtn_code==2 and at_frac_rtn_code!=2 else at_frac    
    at_frac_rtn_code = min(at_frac_rtn_code*vl_frac_rtn_code,2)
    if nom_rtn_code:
        # Case 2 and 3: have composition of the matrix phase and precipitate fraction but not the precipitate comp.
        if mtx_rtn_code and (not prc_rtn_code):
            if wt_frac_rtn_code==2: # Case 2: have wt frac
                prc_wt_comp = (nom_wt_comp-(1.-0.01*wt_frac)*mtx_wt_comp)/(0.01*wt_frac)
                # Need to check for -ve values here.
                if np.any(prc_wt_comp<0.0):
                    # Compute a correction to the wt fraction of the precipitate
                    d_wt_frac = np.nanmin(wt_frac*np.divide(prc_wt_comp,mtx_wt_comp))
                    wt_frac -= d_wt_frac
                    # Recompute the composition wt. %'s
                    prc_wt_comp = (nom_wt_comp-(1.-0.01*wt_frac)*mtx_wt_comp)/(0.01*wt_frac)
            if at_frac_rtn_code==2: # Case 3: have at frac
                prc_at_comp = (nom_at_comp-(1-0.01*at_frac)*mtx_at_comp)/(0.01*at_frac)
                # Need to check for -ve values here.
                if np.any(prc_at_comp<0.0) and correct_if_neg:
                    # Compute a correction to the wt fraction of the precipitate
                    d_at_frac = np.nanmin(at_frac*np.divide(prc_at_comp,mtx_at_comp))
                    at_frac -= d_at_frac
                    # Recompute the composition at. %'s
                    prc_at_comp = (nom_at_comp-(1.-0.01*at_frac)*mtx_at_comp)/(0.01*at_frac)
            if (at_frac_rtn_code==2) ^ (wt_frac_rtn_code==2): # Compute at from wt comp and vv
                prc_wt_comp, prc_at_comp, prc_rtn_code = calc_wt_at_vv(prc_wt_comp,prc_at_comp,masses)
        # Case 4 and 5: have composition of the precipitate phase but not of the matrix (reverse of 2 and 3)
        elif (not mtx_rtn_code) and prc_rtn_code:
            if wt_frac_rtn_code==2: # Case 4: have wt frac
                mtx_wt_comp = (nom_wt_comp-0.01*wt_frac*prc_wt_comp)/(1.-0.01*wt_frac)
                # Need to check for -ve values here.
                if np.any(mtx_wt_comp<0.0):
                    # Compute a correction to the wt fraction of the precipitate
                    d_wt_frac = np.nanmin((100.-wt_frac)*np.divide(mtx_wt_comp,prc_wt_comp))
                    wt_frac += d_wt_frac
                    # Recompute the composition wt. %'s
                    mtx_wt_comp = (nom_wt_comp-0.01*wt_frac*prc_wt_comp)/(1.-0.01*wt_frac)
            if at_frac_rtn_code==2: # Case 5: have at frac
                mtx_at_comp = (nom_at_comp-0.01*at_frac*prc_at_comp)/(1.-0.01*at_frac)
                # Need to check for -ve values here.
                if np.any(mtx_at_comp<0.0):
                    # Compute a correction to the wt fraction of the precipitate
                    d_at_frac = np.nanmin((100.-at_frac)*np.divide(mtx_at_comp,prc_at_comp))
                    at_frac += d_at_frac
                    # Recompute the composition at. %'s
                    mtx_at_comp = (nom_at_comp-0.01*at_frac*prc_at_comp)/(1.-0.01*at_frac)
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
            # In this case the partitioning coefficients can be calculated too.
            K_part_wt = replace_if_new_value(K_part_wt,calc_part_coeff(mtx_wt_comp,prc_wt_comp,x_tol))
            K_part_at = replace_if_new_value(K_part_at,calc_part_coeff(mtx_at_comp,prc_at_comp,x_tol))
            K2_part_wt= calc_part_coeff(nom_wt_comp,prc_wt_comp,x_tol)
            K2_part_at= calc_part_coeff(nom_at_comp,prc_at_comp,x_tol)
            lost_at = 1.-(0.01*at_frac*prc_at_comp+(1.-0.01*at_frac)*mtx_at_comp)/nom_at_comp
            # And also work out the lost amounts.
    if not return_codes:
        return (nom_wt_comp,nom_at_comp,mtx_wt_comp,mtx_at_comp,prc_wt_comp,prc_at_comp,
                wt_frac,at_frac,
                K_part_wt,K_part_at,K2_part_wt,K2_part_at,
                lost_at)
    else:
        return (nom_wt_comp,nom_at_comp,nom_rtn_code,mtx_wt_comp,mtx_at_comp,mtx_rtn_code,
                prc_wt_comp,prc_at_comp,prc_rtn_code,
                wt_frac,wt_frac_rtn_code,at_frac,at_frac_rtn_code,
                K_part_wt,K_part_at,K2_part_wt,K2_part_at,
                lost_at)

# Compute Larson-Miller parameter
def lmp(temp_degC,time_hrs,lmp_param=20,factor=1e-3):
    return factor*(temp_degC+273.15)*(lmp_param+np.log10(time_hrs))

# Process creep data
def process_creep(index,row,stress_bins=None,use_bin_stress=True,count_stresses=False):
    # Have an option to sort stresses according to certain supplied bins
    if np.any(stress_bins):
        stress_vals = np.mean(stress_bins,axis=1)
        n_bins = len(stress_vals)
    else: 
        n_bins = 9
    entry_data = pd.DataFrame(row["Creep life"].values.reshape(n_bins,10)).sort_values(by=[1]).to_numpy()
    output_data = np.full_like(entry_data,np.nan)
    # Get all the stresses in the row and process them
    stresses = []
    for j,entry_col in enumerate(entry_data):
        temp,stress,lmp_1pc,lmp_rupture,t_1pc,t_rupture = tuple(entry_col[:6])
        # Check whether stress entry is non-zero
        if not np.isnan(stress):
            if count_stresses: stresses += [stress]
            # If temp entry is nonzero look for entries for non-LMP data (time data) and convert to LMP
            if not np.isnan(temp):
                if (not np.isnan(t_1pc)) and np.isnan(lmp_1pc): # 1% creep time
                    lmp_1pc = lmp(temp,t_1pc)
                if (not np.isnan(t_rupture)) and np.isnan(lmp_rupture): # Rupture time
                    lmp_rupture = lmp(temp,t_rupture)
            # Identify the correct bin
            if np.any(stress_bins):
                for k,bin_ in enumerate(stress_bins):
                    if bin_[0]<=stress<=bin_[-1]:
                        break
                # If stresses are being binned might want to use the avg. bin value for stress
                if use_bin_stress:
                    stress = stress_vals[k]
            else:
                k = 1*j # Put data into same position as it's in in the input in this case.
            entry_data[j][2] = lmp_1pc
            entry_data[j][3] = lmp_rupture
            output_data[k] = entry_data[j] # Use the output_data variable instead of the entry_data one.
            # NB only write output data in case that a stress value has been found
    if count_stresses:
        return output_data.reshape(10*n_bins), stresses
    else: return output_data.reshape(10*n_bins)
    
# Process the misfit data
def process_misfit(index,row):
    a, a_1_code = check_if_valid(row["γ lattice param"].values[0])
    a_,a_2_code = check_if_valid(row["γ’ lattice param"].values[0])
    misfits = row["RT Lattice misfit"].values
    d1,a_diff,d2 = tuple(misfits)
    d1,d1_code = check_if_valid(d1)
    d2,d2_code = check_if_valid(d2)
    a_diff,a_d_code = check_if_valid(a_diff)
    d1/=100 ; d2/=100 # Originally given as percentages.
    if d2_code==2:
        pass # This is the desired form of the misfit
    else :
        if d1_code==2:
            d2 = d1/(1+0.5*d1)
        elif a_1_code==2 and a_2_code==2:
            d2 = 2*(a_-a)/(a_+a)
        elif a_1_code==2 and a_d_code==2:
            d2 = 2*a_diff/(a_diff+2*a)
        elif a_2_code==2 and a_d_code==2:
            d2 = 2*a_diff/(-a_diff+2*a_)
    d1*=100 ; d2*=100 # Convert back to percentages.
    return a,a_,np.array([d1,a_diff,d2])

# Bin the values of stress used in creep strength tests together using a certain bin size in %.
def bin_stresses(stress_values,bin_size=5):
    # Works with unique or all stresses
    # Bin size in +/- % allowed.
    stress_values.sort()
    bin_bots = []
    bin_tops = []
    bin_vals = []
    count = 0
    bin_counts = []
    bin_top = 0
    for stress in stress_values:
        if stress > bin_top:
            bin_bots += [stress]
            bin_top = (1+0.01*bin_size)*stress
            bin_tops += [bin_top]
            bin_vals += [(1+0.005*bin_size)*stress]
            bin_counts += [count]
            count = 1
        else:
            count += 1
    bin_counts += [count]
    bin_counts.remove(0)
    return np.array([bin_bots,bin_tops]).transpose(), np.array(bin_vals)

#%% Main code
    
# Will make a copy of the dataframe to process
df_proc = df.copy()

# Loop through dataframe w/o modification to get all the stress values
if do_creep_processing:
    stresses = []
    for index, row in df.iterrows():
        new_creep_data, stresses_found = process_creep(index,row,count_stresses=True)
        stresses += stresses_found
    # Bin the stresses that were found
    bins,vals = bin_stresses(stresses)
    bin_number = len(vals)
    # Modify dataframe
    entries_per_test = 10
    lvl_1_names = list(df["Creep life"].columns[:entries_per_test].get_level_values(0))
    current_cols = np.nonzero(df.columns.get_loc("Creep life"))[0]
    orig_bin_num = len(current_cols)//entries_per_test
    for k in range(bin_number-orig_bin_num):
        col_0 = current_cols[-1]+k*entries_per_test+1
        df_proc.insert(int(col_0),("Creep life","Test conditions",'Temp (ºC) .%d' %(k+orig_bin_num)),np.nan)
        df_proc.insert(int(col_0)+1,("Creep life","Test conditions",'Stress (MPa) .%d' %(k+orig_bin_num)),np.nan)
        for l,name in enumerate(lvl_1_names[2:]):
            df_proc.insert(int(col_0+l+2),("Creep life",name,"Unnamed: %d_level_2" %(col_0+l+2)),np.nan)

# Add some new columns for the lost fraction and secondary partitioning coefficient.
for el in els:
    df_proc[("nom/γ’ partitioning ratio","at. %",el)]=np.nan
for el in els:
    df_proc[("Lost elements","at. %",el)]=np.nan

# Process the entire database and write it to the copied database.
for index, row in df_proc.iterrows():
    nom_wt_comp,nom_at_comp,mtx_wt_comp,mtx_at_comp,prc_wt_comp,prc_at_comp,wt_frac,at_frac,K_part_wt,K_part_at,K2_part_wt,K2_part_at,lost_at = process_composition(index,row)
    # Write these values to the copied database
    df_proc.loc[index,("Composition","wt. %")] = nom_wt_comp
    df_proc.loc[index,("Composition","at. %")] = nom_at_comp
    df_proc.loc[index,("γ composition","wt. %")] = mtx_wt_comp
    df_proc.loc[index,("γ composition","at. %")] = mtx_at_comp
    df_proc.loc[index,("γ’ composition","wt. %")] = prc_wt_comp
    df_proc.loc[index,("γ’ composition","at. %")] = prc_at_comp
    df_proc.loc[index,("γ’ fraction","wt. %")] = wt_frac    
    df_proc.loc[index,("γ’ fraction","at. %")] = at_frac
    df_proc.loc[index,("γ’ fraction","vol. %")] = at_frac
    df_proc.loc[index,("γ/γ’ partitioning ratio","wt. %")] = K_part_wt
    df_proc.loc[index,("γ/γ’ partitioning ratio","at. %")] = K_part_at
    df_proc.loc[index,("nom/γ’ partitioning ratio","at. %")] = K2_part_at
    df_proc.loc[index,("Lost elements","at. %")] = lost_at
    # Do the same thing for creep data
    if do_creep_processing:
        new_creep_data = process_creep(index,row,stress_bins=bins)
        df_proc.loc[index,"Creep life"] = new_creep_data
    # ... and for misfit data
    a,a_,misfits = process_misfit(index,row)
    df_proc.loc[index,"γ lattice param"] = a
    df_proc.loc[index,"γ’ lattice param"] = a_
    df_proc.loc[index,"RT Lattice misfit"] = misfits

#%% Write output
    
# Write out the processed database to a csv file.
df_proc.to_csv(output_database)