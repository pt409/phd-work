#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:44:29 2020

@author: Pat Taylor (pt409)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mendeleev import element

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

# Read in a processed dataset.
df = pd.read_csv("../datasets/processing/processed_alloy_database_v2.csv",header=[0,1,2])

def check_if_valid(input_,allow_all_null=False):
    if np.prod("-" == input_) and not allow_all_null:
        return input_, 0 # Invalid input, all entries are -
    else :
        input_ = np.where(input_ == "-",0,input_).astype(np.float64)
        if np.isnan(input_.sum()):
            return input_, 1 # Invalid input, (at least one) empty column
        else :
            return input_, 2 # Valid input

# Get the valid data for a given header
def grab_data(header,df=df,transform_ht=False,warn_if_insuff=False,dat_range=None):
    # Provide header as a tuple
    # transform_ht is a lambda function for x2 variables, temp and time
    ht_num = df.loc[:,"Precipitation heat treatment"].shape[1]
    el_num = df.loc[:,("Composition","at. %")].shape[1]
    out_num = df.loc[:,[header]].shape[1] if len(header) >= 3 else df.loc[:,header].shape[1]
    input_ = np.empty((0,ht_num//2+el_num-1),np.float64) if transform_ht else np.empty((0,ht_num+el_num-1),np.float64)
    output = np.empty((0,out_num),np.float64)
    for index, row in df.iterrows():
        ht, ht_code = check_if_valid(row["Precipitation heat treatment"],allow_all_null=True)
        comp, comp_code = check_if_valid(row["Composition","at. %"])
        comp = comp[1:] # Remove value for Ni as it is not needed
        entry, out_code = check_if_valid(row[header])
        if ht_code==2 and comp_code==2 and out_code==2:
            # If a further constraint has been applied that all of the data points should lie...
            # ...within a certain range of values...
            if dat_range:
                if np.any(entry<dat_range[0]) or np.any(entry>dat_range[1]):
                    continue
            # Might want to apply a transformation to heat treatments: temp, time -> single value.
            if transform_ht:
                ht = np.array([transform_ht(ht_i[0],ht_i[1]) for ht_i in ht.reshape(ht_num//2,2)])
            input_ = np.concatenate((input_,np.array([np.concatenate((ht,comp))])),axis=0)
            output = np.concatenate((output,np.array([entry])),axis=0) if len(header)<3 else np.concatenate((output,np.array([[entry]])),axis=0)
            
    if warn_if_insuff: 
        print("# data points = %d, input dims = %d" % (input_.shape[0],input_.shape[1]))
        if input_.shape[0] <= input_.shape[1]: print("Insufficient # data points (%d) for fitting.\n" % input_.shape[0])
    return np.array(input_), np.array(output)    

# Model of coarsening rate for combining heat treatment times and temperatures.
coarsening_rate = lambda temp,time : time*(273+temp)**(-1/3)
