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
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.kernel_ridge import KernelRidge

from scipy.optimize import minimize

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

def get_microstructure_data(df):
    ms_df = df.copy()
    drop_indices = []
    for index,row in df.iterrows():
        ht, ht_code = check_if_valid(row[("Precipitation heat treatment")],allow_all_null=True)
        if ht_code == 2:
            comp, comp_code = check_if_valid(row[("Composition","at. %")])
            if comp_code == 2:
                frac, frac_code = check_if_valid(row[("γ’ fraction","at. %")])
                if frac_code == 2:
                    prc, prc_code   = check_if_valid(row[("γ’ composition","at. %")])
                    if prc_code == 2:
                        mtx, mtx_code   = check_if_valid(row[("γ composition","at. %")])
                        if mtx_code == 2:
                            continue
        drop_indices += [index]
    ms_df.drop(drop_indices,inplace=True)
    return ms_df

# Model of coarsening rate for combining heat treatment times and temperatures.
coarsening_rate = lambda temp,time : time*(273+temp)**(-1/3)

# Class to define the unique kernel used.
class special_kernel :
    def __init__(self,si,gamma,dim):
        self.tdim = dim+1
        self.S = np.diag(np.append(si,1))
        self.gamma = gamma
        v = np.zeros(self.tdim)
        v[-1] = 1
        v -= np.ones(self.tdim)/np.sqrt(self.tdim)
        self.P = np.identity(self.tdim) - 2*np.outer(v,v)/np.inner(v,v)
        self.Idish = np.r_[np.identity(dim),np.ones((1,dim))]
        
    # Used internally to construct a kernel.
    def construct_trans(self):
        self.M = self.P@self.S@self.P@self.Idish
    
    # Calculate kernelised inner product.
    def kernel(self,x,y):
        return np.exp(-self.gamma*np.linalg.norm(self.M@(x-y),1))
    
    # Update kernel parameters.
    def update_params(self,si,gamma):
        self.S = np.diag(np.append(si,1))
        self.gamma = gamma
        self.construct_trans()
        
    @classmethod
    def setup(cls,si,gamma):
        dim = si.shape[0]
        new_instance = cls(si,gamma,dim)
        new_instance.construct_trans()
        return new_instance
    
# Wrapper for sklearn predictor when there are n models in a cohort and prediction is the mean one.
class cohort_model:
    def __init__(self,n):
        self.n = n # total number of models
        self.n_added = 0 # current number of models
        self.cohort = {}
    
    def add_model(self,model):
        self.cohort[self.n_added] = model
        self.n_added += 1
    
    # As above but for a list
    def add_models(self,models):
        for inst in models: self.add_model(inst)
        
    def predict(self,X):
        if self.n_added == self.n:
            return np.mean([model.predict(X)for model in self.cohort.items()],axis=0)
        else: 
            print("Need {} models and have {} in cohort".format(self.n,self.n_added))
            return None

ms_df = get_microstructure_data(df)
y = ms_df[("γ’ fraction","at. %")].values
X = (ms_df.loc[:,("Composition","at. %")]).drop(["Ni","Hf","Nb"],axis=1).astype(np.float64).values

my_kernel = special_kernel.setup(np.ones(X.shape[1]),0.1)

def train_cohort_model(alpha_j,X,y):
    loo = LeaveOneOut()
    n = loo.get_n_splits(X)
    krr_cohort = cohort_model(n)
    gcv = 0 # Generalised cross validation error
    for train_i,val_i in loo.split(X):
        X_train, X_val = X[train_i], X[val_i]
        y_train, y_val = y[train_i], y[val_i]
        krr_i = KernelRidge(alpha=alpha_j,kernel=my_kernel.kernel) # The kernel ridge regression model.
        krr_i.fit(X_train,y_train)
        dy = y_val - krr_i.predict(X_val)
        gcv += np.dot(dy,dy)[0,0]
        krr_cohort.add_model(krr_i)
    gcv /= n
    return gcv