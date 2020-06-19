#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:44:29 2020

@author: Pat Taylor (pt409)
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from mendeleev import element

#from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score # mean_squared_error
from sklearn.model_selection import LeaveOneOut # cross_validate
from sklearn.kernel_ridge import KernelRidge

from scipy.optimize import minimize

from copy import deepcopy,copy

# Some options.
incl_ht = False
squash_dof = False # Whether to fit "squash" params for composition part of kernel.

# Read in a processed dataset.
df = pd.read_csv("../datasets/processing/processed_alloy_database_v2.csv",header=[0,1,2])
df.set_index("Name")

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

# Get all the entries from the database with "complete" microstructure data.
# "Complete" means complete composiiton + precipitate fraction + phase composition data.
def get_microstructure_data(df,drop_duplicate_comps=False):
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
    if drop_duplicate_comps:
        ms_df = ms_df.loc[ms_df[("Composition","at. %")].drop_duplicates().index]
    return ms_df

# Use in conjunction with get_microstructure to get the X,y data for machine learning purposes.
def get_Xy(df,y_header,drop_els=["Ni","Hf","Nb"],
           min_max=None,drop_na=True,flatten=False,ht=False):
    # Enter header as tuple in case of multiindex
    # drop empty rows
    if drop_na:
        sub_df = df.dropna(subset=[y_header])
    else: sub_df = df.copy()
    # drop rows less/greater than certain min/max values
    if min_max:
        min_, max_ = tuple(min_max)
        if min_: sub_df = sub_df[sub_df[y_header] >= min_] # Min
        if max_: sub_df = sub_df[sub_df[y_header] <= max_] # Max
    y = sub_df[y_header].astype(np.float64).values
    if flatten and len(y.shape) > 1 and y.shape[-1] == 1:
        y = y.flatten()
    X1 = (sub_df.loc[:,("Composition","at. %")]).drop(drop_els,axis=1).astype(np.float64).values
    if ht:
        X0 = sub_df.loc[:,("Precipitation heat treatment")]
        col_order = sorted(X0.columns.tolist(),key = lambda h: h[1])
        X0 = X0[col_order].replace("-",0.0).astype(np.float64).values
        X0[:,:3] += 273.
        X = np.append(X0,X1,axis=1)
    else:
        X = X1
    return X,y

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

# A special kernel to deal with having composition AND heat treatment.
class multi_kernel(special_kernel):
    def __init__(self,si,gamma0,gamma1,comp_dim,ht_dim):
        # ht_dim is the number of heat treatments (temp+time)
        special_kernel.__init__(self,si,gamma0,comp_dim)
        self.gamma1 = gamma1
        self.ht_dim = ht_dim
    
    def kernel(self,x,y):
        T0,t0,x0 = self.split_vector(x)
        T1,t1,x1 = self.split_vector(y)
        return np.exp(-self.gamma *np.linalg.norm(self.M@(x0-x1),1)
                      -self.gamma1*np.dot(T0*t0-T1*t1,T0*t0-T1*t1))
    
    def update_params(self,si,gamma0,gamma1):
        self.S = np.diag(np.append(si,1))
        self.gamma  = gamma0
        self.gamma1 = gamma1
        self.construct_trans()
    
    def split_vector(self,x):
        t = x[:self.ht_dim]
        T = x[self.ht_dim:2*self.ht_dim]
        x0 = x[2*self.ht_dim:]
        return t,T,x0
    
    @classmethod
    def setup(cls,si,gamma0,gamma1,ht_dim):
        comp_dim = si.shape[0]
        new_instance = cls(si,gamma0,gamma1,comp_dim,ht_dim)
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
            return np.mean([model.predict(X) for model in self.cohort.values()],axis=0)
        else: 
            print("Need {} models and have {} in cohort".format(self.n,self.n_added))
            return None
    
    def score(self,X,y):
        return r2_score(y,self.predict(X))
    
# Predict composition of precipitate phase from partitioning coefficients
def predict_phase(models,x_comp,X,elements=["Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]):
    # X is the feature vector (composition or composition & heat treatments)
    # x_comp is the composition of the overall alloy
    N = x_comp.shape[0]
    K = np.empty((N,0))
    f_pred = models["f"].predict(X).reshape(-1,1)
    for el in elements:
        K = np.c_[K,(models[el].predict(X))]
    x_prc = x_comp/((1 - 0.01*f_pred)*K + 0.01*f_pred)
    return x_prc, f_pred

# Process database in order to get all the microstructural data.
ms_df = get_microstructure_data(df,drop_duplicate_comps=(not incl_ht))

# Setup initial kernel.
if incl_ht:
    my_kernel = multi_kernel.setup(np.ones(9),0.1,0.1,3)
else:
    my_kernel = special_kernel.setup(np.ones(9),0.1) # 9 is dimensionality of composition data.

def train_cohort_model(alpha_j,X,y,return_model=False):
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
        gcv += np.dot(dy,dy)/(dy.shape[0])
        krr_cohort.add_model(krr_i)
    gcv /= n
    if return_model:
        return krr_cohort # This way the model can be accessed without returning it.
    else:
        return gcv

#result = minimize(train_cohort_model,[0.1],args=(X,y))

# Get all the data for precipitate fraction and g/g' partitioning coeff. 
# Store as tuples in a dict for fast access.
elements = ["Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]
ml_data_dict = {}
for el in elements:
    part_coeff_header = ("γ/γ’ partitioning ratio","at. %",el)
    if not part_coeff_header in ms_df:
        part_coeff_header = ("γ/γ’ partitioning ratio","at. %",el+" ")
    ml_data_dict[el] = get_Xy(ms_df,part_coeff_header,min_max=[0.0,100.0],ht=incl_ht)
ml_data_dict["f"] = get_Xy(ms_df,("γ’ fraction","at. %"),drop_na=False,flatten=True,ht=incl_ht)
output_head = "        \t"+"\t".join(elements + ["f"])

# Target data
X_ms = ml_data_dict["f"][0]
x_comp = (ms_df.loc[:,("Composition","at. %")]).drop(["Ni","Hf","Nb"],axis=1).astype(np.float64).values
x_prc_target = (ms_df.loc[:,("γ’ composition","at. %")]).drop(["Ni","Hf","Nb"],axis=1).astype(np.float64).values
f = ml_data_dict["f"][1].reshape(-1,1)
N = x_comp.shape[0]

# Kernel ridge model for each microstructural property.
next_alpha = {ms_prop:0.1 for ms_prop in ml_data_dict.keys()}
models = {}
opt_models = {}
best_error = np.inf
# Define as a function for use with minimiser:
def calc_microstruc_error(sij,v=1):
    # v is the verbosity, 0, 1 or 2 (most verbose).
    # First update the kernel with new parameters:
    global my_kernel
    if incl_ht:
        if squash_dof:
            gamma, sij = tuple(sij[-2:]), sij[:-2]
            sij = np.insert(sij,0,1.0) # Don't need to stretch in every possible direction.
        else:
            gamma = tuple(sij[-2:])
            sij = np.ones(9)
    else:
        if squash_dof:
            gamma, sij = tuple(sij[-1:]), sij[:-1]
            sij = np.insert(sij,0,1.0) # Don't need to stretch in every possible direction.
        else:
            gamma = tuple(sij[-1:])
            sij = np.ones(9)
    my_kernel.update_params(sij,*gamma)    
    if v >= 1:
        print("s_ii =\t"+("\t".join("{:.5f}".format(_) for _ in sij.tolist())))
        print("gamma =\t"+("\t".join("{:.6f}".format(_) for _ in gamma)))
        print("Beginning to fit kernel ridge models for microstructural properties.")
    # Loop through each microstructural property and train optimal krr model using cv.
    # Store some values calculated for output purposes.
    lambda_output = "lambda =\t"
    score_output =  "R^2    =\t"
    error_output =  "CV err =\t"
    for ms_prop, ms_data in ml_data_dict.items():
        
        if v>=1: print("Microstructural property {} ...".format(ms_prop))
        result = minimize(train_cohort_model,
                          next_alpha[ms_prop],
                          args=ms_data,
                          method="L-BFGS-B",
                          bounds=[(1.e-9,None)],
                          options={"ftol":3.e-5,
                                   "gtol":1.e-5,
                                   "eps":1.e-5})
        opt_alpha = result.x
        next_alpha[ms_prop] = opt_alpha
        krr_model = train_cohort_model(opt_alpha,*ms_data,return_model=True)
        models[ms_prop] = krr_model
        if v >= 2:
            lambda_output += "{:.6f}\t".format(result.x[0])
            try: 
                error_output += "{:.6f}\t".format(result.fun)
            except TypeError:
                return result
            score_output += "{:.5f}\t".format(krr_model.score(*ms_data))
    if v >= 1: print("Done!\n")
    if v >= 2:
        print(output_head)
        print(score_output)
        print(error_output)
        print(lambda_output)
    # Calculate the predicted phase composition. 
    N = x_comp.shape[0]
    K = np.empty((N,0))
    f_pred = models["f"].predict(X_ms).reshape(-1,1)
    for el in elements:
        K = np.c_[K,(models[el].predict(X_ms))]
    mu = 1.0 # Weighting of overall precipitate fraction in the score.
    error = (0.5 * mu * 1.e-4*(f - f_pred)**2 
             + 1.e-8*((f_pred * x_comp/((1 - 0.01*f_pred)*K + 0.01*f_pred) - f*x_prc_target)**2).sum()).sum(axis=0)
    error /= N
    if v >= 1:
        error_0 = (0.5 * 1.e-4*(f - f.mean(axis=0))**2 
                   + 1.e-8*((f.mean(axis=0) * x_prc_target.mean(axis=0) - f*x_prc_target)**2).sum()).sum(axis=0)/N
        score = 1.0 - error/error_0
        print("\nMicrostructural error = {:.6f} score = {:.5f}\n".format(error[0],score[0]))
    # Store models if the error is lowest yet.
    global best_error
    global opt_models
    if error < best_error:
        opt_models = deepcopy(models)
        best_error = copy(error)
    # Finally return error
    return error

# Now minimise the microstructural error over the kernel parameters.
v = 2 # verbosity
if incl_ht:
    if squash_dof:
        sij_init = np.ones(10)
        sij_init[-2] = 0.15 # Represents the gamma parameters.
        sij_init[-1] = 1.e-3 # The gamma1 parameter.
    else:
        sij_init = 0.1*np.ones(2)
        sij_init[-1] = 1.e-3 # The gamma1 parameter.
else:
    if squash_dof:
        sij_init = np.ones(9)
        sij_init[-1] = 0.15 # Represents the gamma parameter.
    else:
        sij_init = 0.1*np.ones(1)
result = minimize(calc_microstruc_error,
                  sij_init,
                  args=(v,),
                  method="BFGS",
                  options={"gtol":1.e-4,
                           "eps":1.e-3})