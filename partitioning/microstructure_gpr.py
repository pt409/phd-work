#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:44:29 2020

@author: Pat Taylor (pt409)
"""

import numpy as np
import pandas as pd
#import pickle

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
#from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform

import matplotlib.pyplot as plt

#from itertools import product

from copy import deepcopy,copy

import configparser
import sys

# Function to parse list/tuple-like things in config file:
def parse_listlike(list_string):
    flag = 0
    py_groups = []
    for word in groups.split(","):
        if word[0]=="(":
            flag = 1
            py_groups += [(int(word[1:]),)]
        elif word[-1]==")":
            py_groups[-1] += (int(word[:-1]),)
            flag = 0
        elif flag==1:
            py_groups[-1] += (int(word),)
        else:
            py_groups += [int(word)]
    return py_groups
    
# Some options.
# Uses a configparser file (.ini structure).
config_default = "test"
config = configparser.ConfigParser()
config.read("microstructure_gpr.input")
if len(sys.argv) > 1:
    if sys.argv[1] in config.sections():
        config_type = sys.argv[1]
    else: config_type = config_default
else: config_type = config_default
my_config = config[config_type]
# Read config file in:
incl_ht = True # Always include ht in this version of the code.
#opt_models_pkl_0 = config[config_type].get("output_models")
database =          my_config.get("database")
elements =          my_config.get("elements")
elements = elements.split(",")
n_els = len(elements)
n =                 my_config.getint("feature_vector_size")
n_ht =              my_config.getint("ht_features")
ht_range = [0,n_ht] ; comp_range = [n_ht,n]
# Train/test split related stuff
test_frac =         my_config.getfloat("test_pc")/100.0
seed =              my_config.getint("seed")
n_fold_testing =    my_config.getboolean("n_fold_testing")
n_folds =           my_config.getint("n_folds")
# Kernel related stuff
standardise_ht =    my_config.getboolean("standardise_ht")
standardise_comp =  my_config.getboolean("standardise_comp")
comp_kernel_0 =     my_config.get("comp_kernel_0")
comp_kernel_1 =     my_config.get("comp_kernel_1")
ht_kernel_0 =       my_config.get("ht_kernel_0")
ht_kernel_1 =       my_config.get("ht_kernel_1")
alpha_noise =       my_config.getfloat("kernel_noise")
groups =            my_config.get("projection_groups")
groups = parse_listlike(groups)
num_groups =        my_config.getint("projection_rank")
# Verbosity of output
v =                 my_config.getint("verbosity")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PROCESSING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Read in a processed dataset.
df = pd.read_csv(database,header=[0,1,2])
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

# Get all the entries from the database with "complete" microstructure data.
# "Complete" means complete composiiton + precipitate fraction + phase composition data.
def get_microstructure_data(df,drop_duplicate_comps=False,shuffle_seed=None):
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
    # Shuffle database and select a specified fraction of it:
    ms_df=ms_df.sample(frac=1.,random_state=shuffle_seed).reset_index(drop=True)
    return ms_df

# Use in conjunction with get_microstructure to get the X,y data for machine learning purposes.
def get_Xy(df,y_header,drop_els=["Ni","Hf","Nb"],
           min_max=None,drop_na=True,flatten=False,ht=False,log_y=False):
    # Enter header as tuple in case of multiindex
    # drop rows less/greater than certain min/max values
    if drop_na:
        sub_df = df.dropna(subset=y_header)
    else:
        sub_df = df.copy()
    if min_max:
        min_, max_ = tuple(min_max)
        if isinstance(min_,float): 
            condition_0 = (sub_df != False)
            condition = sub_df[y_header].astype("float64") > min_ # Min
            condition_0.update(condition)
            sub_df = sub_df[condition_0].dropna(subset=y_header)
        if isinstance(max_,float): 
            condition_0 = (sub_df != False)
            condition = sub_df[y_header].astype("float64") < max_ # Max
            condition_0.update(condition)
            sub_df = sub_df[condition_0].dropna(subset=y_header)
    # Now drop empty rows
    # Start getting data here:
    y = sub_df.loc[:,y_header].astype("float64").values
    if flatten and len(y.shape) > 1 and y.shape[-1] == 1:
        y = y.flatten()
    if log_y:
        y = np.log(y)
    X1 = 0.01*(sub_df.loc[:,("Composition","at. %")].drop(drop_els,axis=1).astype("float64").values)
    if ht:
        X0 = sub_df.loc[:,("Precipitation heat treatment")]
        col_order = sorted(X0.columns.tolist(),key = lambda h: h[1])
        X0 = X0[col_order].replace("-",0.0).astype(np.float64).values
        X0[:,:3] += 273.
        X = np.append(X0,X1,axis=1)
    else:
        X = X1
    return X,y

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SCALER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Based on sklearn.preprocessing.StandardScaler
class PartScaler():
    def __init__(self, scale_range=None, copy_=True,with_mean=True,with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy_ = copy_
        self.range_ = scale_range
        
    def _reset(self):
        if hasattr(self,'scale_'):
            del self.scale_
            del self.offset_
            
    def fit(self,X):
        self._reset()
        if self.with_mean:
            if self.range_:
                self.offset_ = np.zeros(X.shape[1])
                self.offset_[self.range_[0]:self.range_[1]] = np.mean(X[:,self.range_[0]:self.range_[1]],axis=0)
            else:
                self.offset_ = np.mean(X,axis=0)                
        else: 
            self.offset_ = 0.0
        
        if self.with_std:
            if self.range_:
                self.scale_ = np.ones(X.shape[1])
                self.scale_[self.range_[0]:self.range_[1]] = np.std(X[:,self.range_[0]:self.range_[1]],axis=0)
            else:
                self.scale_ = np.std(X,axis=0)
            self.scale_ = np.where(self.scale_==0.0,1.0,self.scale_)
        else:
            self.scale_ = 1.0
        return self
    
    def transform(self,X,copy_=None):
        copy_ = copy_ if copy_ is not None else self.copy_
        if copy_:
            X = X.copy()
        X -= self.offset_
        X /= self.scale_
        return X
        
    def inverse_transform(self,X,copy_=None):
        copy_ = copy_ if copy_ is not None else self.copy_
        if copy_:
            X = X.copy()
        X *= self.scale_
        X += self.offset_
        return X
    
    def fit_transform(self,X,copy_=None):
        self.fit(X)
        return self.transform(X,copy_)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KERNELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modification of base RBF class
class L2RBF(gp.kernels.RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 dims=15,dim_range=None,comp=True):
        super(L2RBF,self).__init__(length_scale,length_scale_bounds)
        # Matrix used to transform vectors in call.
        self.dims = dims
        self.dim_range = dim_range
        self.comp = comp
        self.constr_trans()
        
    def constr_trans(self):
        A = np.eye(self.dims)
        if self.dim_range: 
            A = A[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            A = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],A]
        A = A.T # Use transpose since vectors are represented by rows not columns.
        self.A = A
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = gp.kernels._check_length_scale(X@self.A, self.length_scale)
        if Y is None:
            dists = pdist(X@self.A / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X@self.A / length_scale, Y@self.A / length_scale,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty(((X@self.A).shape[0], (X@self.A).shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = ((X@self.A)[:, np.newaxis, :] - (X@self.A)[np.newaxis, :, :])**2 \
                    / length_scale
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

# Mainly borrowed from RBF class
class L1RBF(L2RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 dims=15,dim_range=None,comp=True):
        super(L1RBF,self).__init__(length_scale,length_scale_bounds,
                 dims,dim_range,comp)
                
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = gp.kernels._check_length_scale(X@self.A, self.length_scale)
        if Y is None:
            dists = pdist(X@self.A / length_scale, metric='cityblock')
            K = np.exp(-1. * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X@self.A / length_scale, Y@self.A / length_scale,
                          metric='cityblock')
            K = np.exp(-1. * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty(((X@self.A).shape[0], (X@self.A).shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = np.abs((X@self.A)[:, np.newaxis, :] - (X@self.A)[np.newaxis, :, :]) \
                    / length_scale
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

# "Physical" kernel for the heat treatment part of feature vector.
class physRBF(gp.kernels.RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 dims=15,dim_range=None):
        super(physRBF,self).__init__(length_scale,length_scale_bounds)
        self.dims = dims
        self.dim_range = dim_range
        # Matrix used to transform vectors in call.
        if self.dim_range: 
            ht_dims = self.dim_range[1]-self.dim_range[0]
        else: ht_dims = self.dims
        if ht_dims%2 != 0:
            raise ValueError(
                "Need an even number of dimensions")
        else: ht_dims //=2
        self.M = np.zeros((self.dims,self.dims,ht_dims))
        for k in range(ht_dims):
            self.M[k,k+ht_dims,k] = 0.5
            self.M[k+ht_dims,k,k] = 0.5
              
    def __call__(self, X, Y=None, eval_gradient=False):
        # Convert input vector to representation of physical HT part.
        X = np.einsum("li,lj,ijk->lk",X,X,self.M)
        X = np.atleast_2d(X)
        length_scale = gp.kernels._check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            # COnvert 2nd input vector to representation of physical HT part.
            Y = np.einsum("li,lj,ijk->lk",Y,Y,self.M)
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2 \
                    / length_scale
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K
        
# RBF classes with projection of composition onto a smaller subspace
class subspace_L2RBF(L2RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 groups=np.arange(15),num_groups=15,
                 dims=15,dim_range=None,comp=True):
        self.groups = groups
        self.num_groups = num_groups
        super(subspace_L2RBF,self).__init__(length_scale,length_scale_bounds,
                 dims,dim_range,comp)
        
    def constr_trans(self):
        # Same part of transformation as above (convert to composition)
        Ilike = np.eye(self.dims)
        if self.dim_range: 
            Ilike = Ilike[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            Ilike = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],Ilike]        # Subspace projection part of matrix
        A = np.zeros([self.num_groups,len(self.groups)])
        for col,group in enumerate(self.groups):
            A[group][col]=1.
        self.A = (A @ Ilike).T # Use transpose since vectors are represented by rows not columns.
    
class subspace_L1RBF(L1RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 groups=np.arange(15),num_groups=15,
                 dims=15,dim_range=None,comp=True):
        self.groups = groups
        self.num_groups = num_groups
        super(subspace_L1RBF,self).__init__(length_scale,length_scale_bounds,
                 dims,dim_range,comp)
        
    def constr_trans(self):
        # Same part of transformation as above (convert to composition)
        Ilike = np.eye(self.dims)
        if self.dim_range: 
            Ilike = Ilike[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            Ilike = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],Ilike]        # Subspace projection part of matrix
        A = np.zeros([self.num_groups,len(self.groups)])
        for col,group in enumerate(self.groups):
            A[group][col]=1.
        self.A = (A @ Ilike).T # Use transpose since vectors are represented by rows not columns.
        
# Allow projections but also allow mixing of groups.
class subspace_mixing_L2RBF(subspace_L2RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 mix_vals = np.array([0.5]),mix_val_bounds = (1.e-6,1.0),
                 groups=np.arange(15),num_groups=15,
                 dims=15,dim_range=None,comp=True):
        self.mix_vals = mix_vals
        self.mix_val_bounds = mix_val_bounds
        super(subspace_mixing_L2RBF,self).__init__(length_scale,length_scale_bounds,
                                                   groups,num_groups,dims,dim_range,comp)
    
    @property
    def hyperparameter_mix_vals(self):
        return gp.kernels.Hyperparameter("mix_vals","numeric",
                                         self.mix_val_bounds,
                                         len(self.mix_vals))
    
    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], mix_vals=[{2}])".format(
                self.__class__.__name__, 
                ", ".join(map("{0:.3g}".format,self.length_scale)),
                ", ".join(map("{0:.3g}".format,self.mix_vals)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g}, mix_vals=[{2}])".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0],
                ", ".join(map("{0:.3g}".format,self.mix_vals)))
        
    def constr_trans(self):
        # Same part of transformation as above (convert to composition)
        Ilike = np.eye(self.dims)
        mixed_groups_only = np.empty_like(self.mix_vals,dtype="int",shape=(self.mix_vals.shape[0],2)) # store for later
        mixed_el_pos = np.empty_like(self.mix_vals,dtype="int")
        if self.dim_range: 
            Ilike = Ilike[self.dim_range[0]:self.dim_range[1],:]
        if self.comp: 
            Ilike = np.r_[[np.append(np.zeros(self.dim_range[0]),np.ones(self.dims-self.dim_range[0]))],Ilike]
        # Subspace projection part of matrix
        A = np.zeros([self.num_groups,len(self.groups)])
        mix_val_pos = 0
        for col,group in enumerate(self.groups):
            if np.size(group)==1:
                A[group][col]=1.
            else: # Mixed groups
                mixed_groups_only[mix_val_pos] = group
                mixed_el_pos[mix_val_pos] = col
                for i,group_i in enumerate(group):
                    if i>0:
                        A[group_i][col] = self.mix_vals[mix_val_pos]
                    else:
                        A[group_i][col] = 1.-self.mix_vals[mix_val_pos]
                mix_val_pos += 1
        self.mixed_groups = mixed_groups_only
        self.mixed_el_pos = mixed_el_pos
        self.A = A.T # Use transpose since vectors are represented by rows not columns.
        self.Ilike = Ilike.T
        
    def update_trans(self):
        self.A[self.mixed_el_pos,self.mixed_groups[:,0]] = 1. - self.mix_vals
        self.A[self.mixed_el_pos,self.mixed_groups[:,1]] = self.mix_vals
        
    def compute_gradient(self,X,X_pr,K,gradient):
        for ab,i,mix_val in zip(self.mixed_groups,self.mixed_el_pos,self.mix_vals):
            a,b = ab
            grad_i = squareform(pdist(X,lambda x,y: x[i]-y[i])*pdist(X_pr,lambda x,y: x[b]-x[a]-y[b]+y[a]))
            grad_i *= -mix_val*K/(self.length_scale**2)
            grad_i = grad_i[:,:,np.newaxis]
            gradient = np.dstack((gradient,grad_i))
        return gradient
    
    def __call__(self, X, Y=None, eval_gradient=False):
        self.update_trans()
        X = np.atleast_2d(X) @ self.Ilike # Change the vector to one representing composition incl. base element
        X_pr = X @ self.A # X projected onto subspace
        length_scale = gp.kernels._check_length_scale(X_pr, self.length_scale)
        if Y is None:
            dists = pdist(X_pr / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y = Y @ self.Ilike
            Y_pr = Y @ self.A
            dists = cdist(X_pr/ length_scale, Y_pr / length_scale,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X_pr.shape[0], X_pr.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                length_scale_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                length_scale_gradient = (X_pr[:, np.newaxis, :] - X_pr[np.newaxis, :, :])**2 \
                    / length_scale
                length_scale_gradient *= K[..., np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
        else:
            return K
        
# Allow projections but also allow mixing of groups.
class subspace_mixing_L1RBF(subspace_mixing_L2RBF):
    def __init__(self,length_scale=1.0,length_scale_bounds=(1.e-5,1.e5),
                 mix_vals = np.array([0.5]),mix_val_bounds = (1.e-6,1.0),
                 groups=np.arange(15),num_groups=15,
                 dims=15,dim_range=None,comp=True):
        super(subspace_mixing_L1RBF,self).__init__(length_scale,length_scale_bounds,
                                                   mix_vals,mix_val_bounds,
                                                   groups,num_groups,dims,dim_range,comp)
    
    def compute_gradient(self,X,X_pr,K,gradient):
        for ab,i,mix_val in zip(self.mixed_groups,self.mixed_el_pos,self.mix_vals):
            a,b = ab
            grad_i = squareform(pdist(X,lambda x,y: x[i]-y[i]) \
                                *pdist(X_pr,lambda x,y: np.sign(x[b]-y[b]) - np.sign(x[a]-y[a])))
            grad_i *= -mix_val*K/self.length_scale
            grad_i = grad_i[:,:,np.newaxis]
            gradient = np.dstack((gradient,grad_i))
        return gradient
    
    def __call__(self, X, Y=None, eval_gradient=False):
        self.update_trans()
        X = np.atleast_2d(X) @ self.Ilike # Change the vector to one representing composition incl. base element
        X_pr = X @ self.A # X projected onto subspace
        length_scale = gp.kernels._check_length_scale(X_pr, self.length_scale)
        if Y is None:
            dists = pdist(X_pr / length_scale, metric='cityblock')
            K = np.exp(-dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y = Y @ self.Ilike
            Y_pr = Y @ self.A
            dists = cdist(X_pr/ length_scale, Y_pr / length_scale,
                          metric='cityblock')
            K = np.exp(-dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X_pr.shape[0], X_pr.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                length_scale_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                length_scale_gradient = np.abs(X_pr[:, np.newaxis, :] - X_pr[np.newaxis, :, :]) \
                    / length_scale
                length_scale_gradient *= K[..., np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
        else:
            return K

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODEL CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# These 2 classes do all the heavy lifting of fitting models and making predictions
class sub_model():
    def __init__(self,gpr_model,scaler,is_element=False):
        self.gpr = deepcopy(gpr_model)
        if scaler:
            self.scaler = deepcopy(scaler)
        else: self.scaler = None
        # If the sub_model represents a certain element then we want to exclude it from predictions.
        self.is_element = is_element
        self.fitted_flag = False
        self.train_X = None
        self.train_y = None
        
    def fit(self,X,y):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.gpr.fit(X,y)
        self.fitted_flag = True
        self.train_X = X
        self.train_y = y
    
    def predict(self,X,return_std=True,log_y=True):
        if self.fitted_flag:
            if self.scaler:
                X = self.scaler.transform(X)
            if return_std:
                y,y_std = self.gpr.predict(X,return_std=return_std)
                if log_y:
                    y = np.exp(y)
                    #y_std *= y # Don't need this, return fractional uncertainty.
                else:
                    y_std /= y # Fractional uncertainty.
                if self.is_element:
                    y = np.where(X[:,self.is_element]==0.0,np.nan,y)
                    y_std = np.where(X[:,self.is_element]==0.0,np.nan,y_std)
                return y,y_std
            else:
                y = self.gpr.predict(X,return_std=return_std)
                if log_y:
                    y = np.exp(y)
                if self.is_element:
                    y = np.where(X[:,self.is_element]==0.0,np.nan,y)
                return y
        else: raise ValueError(
            "Model has not yet been fitted.")
        
    # Function to print mixing values from a fitted kernel
    def get_mixing_vals(self,elements):
        all_params = self.gpr.kernel_.get_params()
        for param_name,param_value in all_params.items():
            if "mix_vals" in param_name:
                kernel_component = self.gpr.kernel_
                for word in param_name.split("__"):
                    if "k1" in word:
                        kernel_component = kernel_component.k1
                    elif "k2" in word:
                        kernel_component = kernel_component.k2
                trans = kernel_component.A.T
                for row in trans:
                    print(" + ".join([" ".join(_) for _ in np.transpose([list(map("{:.4f}".format,row)),elements])]))


class model():
    def __init__(self,elements,kernel,alpha,seed,
                 standardise_comp,standardise_ht,
                 comp_range,ht_range,
                 n_restarts_optimizer=2,
                 phases=2,kernel_pr_mixing=False):
        self.proto_kernel = kernel
        self.elements = elements
        self.alpha = alpha
        self.scale_comp = standardise_comp ; self.scale_ht = standardise_ht
        self.comp_range = comp_range ; self.ht_range = ht_range
        self.n_restarts_optimizer = n_restarts_optimizer
        self.phases = phases
        self.mixing = kernel_pr_mixing
        # Run setup of various sub_models
        self.__setup_sub_models()
        
        
    def __setup_sub_models(self):
        # Setup a generic scaler
        if self.scale_ht and self.scale_comp:
            scaler = PartScaler(with_mean=False)
        elif self.scale_ht:
            scaler = PartScaler(self.ht_range,with_mean=False)
        elif self.scale_comp:
            scaler = PartScaler(self.comp_range,with_mean=False)
        else:
            scaler = None
        # Setup a generic Gaussian process regression model
        gpr = GaussianProcessRegressor(kernel=self.proto_kernel,
                                           normalize_y=True,
                                           random_state=seed,
                                           alpha=self.alpha,
                                           n_restarts_optimizer=self.n_restarts_optimizer)
        # Add all the relevant sub-models.
        self.sub_models = {}
        for ind,el in enumerate(self.elements):
            self.sub_models[el] = {}
            # Pass this value to sub_model to work out which elements to exclude in prediction.
            if ind>0: # Assume base element is always included.
                element_ind = ind - len(elements)
            else:
                element_ind = False
            for phase in range(self.phases):
                part_coeff_name = "nom:{:}".format(phase+1) if phase==0 else "{:}:{:}".format(phase,phase+1)
                self.sub_models[el][part_coeff_name] = sub_model(gpr,scaler,element_ind)
        self.sub_models["f"] = {}
        for phase in range(1,self.phases):
            self.sub_models["f"]["f{:}".format(phase)] = sub_model(gpr,scaler)
                
    def matching_data_struct(self):
        # Return a data structure matching that of the model
        structure = {}
        for el in self.elements:
            structure[el] = {}
            for phase in range(self.phases):
                part_coeff_name = "nom:{:}".format(phase+1) if phase==0 else "{:}:{:}".format(phase,phase+1)
                structure[el][part_coeff_name] = None
        structure["f"] = {}
        for phase in range(1,self.phases):
            structure["f"]["f{:}".format(phase)] = None
        return structure
        
    def fit_sub_models(self,Xy_data,verbosity=1):
        # Assume matching data structure, containing tuples or lists containing X,y.
        for ms_ft,sub_fts in self.sub_models.items():
            for sub_ft,model in sub_fts.items():
                model.fit(*Xy_data[ms_ft][sub_ft])
                # Print some output from the model.
                if verbosity >= 1:
                    print("Fitting model for "+ms_ft+" phase "+sub_ft+" feature...")
                    if verbosity >= 2 and self.mixing:
                        print("Pseudo-element representation for model:")
                        model.get_mixing_vals(self.elements)
                    print(120*"-")
    
    def sub_predict(self,X_data,
                    data_structured=True,log_y=True,
                    return_std=True,have_y_data=False):
        # This function makes predicitions for all the sub_models but doesn't do anything with them.
        if data_structured: # dict
            predictions = self.matching_data_struct()
        else: # arra
            to_fill = np.empty((X_data.shape[0],len(self.elements)))
            predictions = {("nom:{:}".format(phase+1) if phase==0 else "{:}:{:}".format(phase,phase+1)):
                           {"val":to_fill.copy(),"std":to_fill.copy()} for phase in range(self.phases)}
            to_fill = np.empty((X_data.shape[0],1))
            for phase in range(self.phases): predictions["f{:}".format(phase)] = {"val":to_fill.copy(),"std":to_fill.copy()}
        # Assume matching data structure, containing tuples or lists containing X,y, or just X, if data is structured.
        for i,(ms_ft,sub_fts) in enumerate(self.sub_models.items()):
            for a,(sub_ft,sub_model) in enumerate(sub_fts.items()):
                if data_structured:
                    if have_y_data:
                        predictions[ms_ft][sub_ft] = sub_model.predict(X_data[ms_ft][sub_ft][0],
                                                                       return_std=return_std,log_y=log_y)
                    else:
                        predictions[ms_ft][sub_ft] = sub_model.predict(X_data[ms_ft][sub_ft],
                                                                       return_std=return_std,log_y=log_y)
                else:
                    y,y_std = sub_model.predict(X_data,return_std=return_std,log_y=log_y)
                    if i<len(self.elements):
                        predictions[sub_ft]["val"][:,i] = y
                        predictions[sub_ft]["std"][:,i] = y_std
                    else:
                        predictions[sub_ft]["val"][:,0] = y
                        predictions[sub_ft]["std"][:,0] = y_std
        return predictions
    
    # Currently this only handles two phases alloys.
    def predict(self,X,return_std=True,log_y=True):
        # Here X should be just an array.
        # x is composition part of X
        N = X.shape[0]
        x = np.zeros((N,len(self.elements)))
        x[:,1:] = X[:,self.comp_range[0]:self.comp_range[1]]
        x[:,0]  = 1.-x.sum(axis=1)
        # Different f models are stored for comparison
        f_all = np.empty((len(self.sub_models.keys()),N))
        f_std = f_all.copy()
        # Predicitions for each sub-component of the microstrucutre using gpr model.
        sub_predictions = self.sub_predict(X,data_structured=False,return_std=True,log_y=True)
        f_0 = sub_predictions["f1"]["val"]
        f_std_0 = sub_predictions["f1"]["std"]
        f_all[0,:] = f_0
        f_std[0,:] = f_std_0
        # Calculate microstructural features for each alloy, and associated errors.
        for i,el in enumerate(self.elements):
            K1,K1_std = sub_predictions[el]["nom:1"]
            K2,K2_std = sub_predictions[el]["1:2"]
            # Calculate fraction according to this model.
            f_i = (K2-K1)/(1.-K1)
            f_i_std = np.sqrt(((K2-1.)*K1_std)**2+((K1-1.)*K2_std)**2)/(1.-K1)**2
            f_all[i+1,:] = f_i
            f_std[i+1,:] = f_i_std
            # Calculate x_i_prc according to 1st part. coeff. model
            x_i = x[:,i]
            x_i_prc_mod1 = x_i/np.abs(f_0+(1.-f_0)*K1)
            x_i_std_mod1 = np.sqrt(((1.-K1)*f_std_0)**2 + ((1.-f_std_0)*K1_std)**2)*x_i_prc_mod1**2/x_i
            # Calculate x_i_prc according to 2nd part. coeff. model
            x_i_prc_mod2 = x_i/K2
            x_i_std_mod2 = (x_i/K2**2)*K2_std
        # Now choose best model for precipitate fraction
        f_std_frac = np.abs(f_std/f_all) # use fractional error
        best_locs = np.where(f_std_frac == np.nanmin(f_std_frac,axis=0))
        best_locs = (best_locs[0][best_locs[1].argsort()],np.sort(best_locs[1]))
        best_models = best_locs[0] # Store which models were best for which data points.
        f_best = f_all[best_locs]
        f_best_std = f_std[best_locs] # Want absolute errors here
        return f_best,f_best_std,best_models
    
    def Predict(self,X,lambda_=0.0):
        # Here X should be just an array.
        # x is composition part of X
        N = X.shape[0]
        x = np.zeros((N,len(self.elements)))
        x[:,1:] = X[:,self.comp_range[0]:self.comp_range[1]]
        x[:,0]  = 1.-x.sum(axis=1)
        # Predicitions for each sub-component of the microstrucutre using gpr model.
        sub = self.sub_predict(X,data_structured=False,return_std=True,log_y=True)
        f_dir = sub["f1"]["val"].reshape(-1)
        f_dir_std = sub["f1"]["std"].reshape(-1)
        # Phase 1 is the gamma' phase, phase 2 is the gamma phase.
        x_1 = x/sub["nom:1"]["val"]
        x_2 = x_1*sub["1:2"]["val"]
        # Note these are fractional errors
        x_1_stdf = sub["nom:1"]["std"]
        x_2_stdf = np.sqrt(sub["nom:1"]["std"]**2+sub["1:2"]["std"]**2)
        x_1_std  = x_1_stdf*x_1
        x_2_std  = x_2_stdf*x_2
        # Work out where max values occur.
        calc_j_1 = np.nanargmax(x_1_std,axis=1)
        calc_j_2 = np.nanargmax(x_2_std,axis=1)
        for a,(j_1,j_2) in enumerate(zip(calc_j_1,calc_j_2)):
            # Find new, calculated element.
            x_1[a,j_1] = 0.0 ; x_1[a,j_1] = 1.-np.nansum(x_1[a])
            x_2[a,j_2] = 0.0 ; x_2[a,j_2] = 1.-np.nansum(x_2[a])
            # Find new, calculated uncertainty.
            x_1_std[a,j_1] = 0.0 ; x_1_std[a,j_1] = np.sqrt(np.nansum(x_1_std[a]**2))
            x_2_std[a,j_2] = 0.0 ; x_2_std[a,j_2] = np.sqrt(np.nansum(x_2_std[a]**2))
        # Calculate f to compare to direct value
        f_calc = np.nansum((x_1-x_2)*(x-x_2)/x_2_std**2,axis=1)\
            /(np.nansum((x_1-x_2)**2/x_2_std**2,axis=1)+lambda_)
        f_calc_std = np.std(np.nansum(((x-2*f_calc[:,np.newaxis]*x_1+(2*f_calc[:,np.newaxis]-1)*x_2)**2*(x_1_std/x_2_std)**2+\
                                       ((1-2*f_calc[:,np.newaxis])*x_1+2*(f_calc[:,np.newaxis]-1)*x_1+x)**2)/x_2_std**2,axis=1))\
            /np.abs(np.nansum((x_1-x_2)**2/x_2_std**2,axis=1)+lambda_)
        f_codes = f_calc_std < f_dir_std
        f_std = np.fmin(f_calc_std,f_dir_std)
        f = f_codes*f_calc + (~f_codes)*f_dir
        model_codes = np.array([f_codes,calc_j_1,calc_j_2]).T
        return f,f_std,x_1,x_1_std,x_2,x_2_std,model_codes
            
    def predict_(self,X):
        # Here X should be just an array.
        # x is composition part of X
        N = X.shape[0]
        x = np.zeros((N,len(self.elements)))
        x[:,1:] = X[:,self.comp_range[0]:self.comp_range[1]]
        x[:,0]  = 1.-x.sum(axis=1)
        # Predicitions for each sub-component of the microstrucutre using gpr model.
        sub = self.sub_predict(X,data_structured=False,return_std=True,log_y=True)
        # Certain values that will keep coming up in calc.
        f_dir = sub["f1"]["val"]
        f_dir_std = sub["f1"]["std"]
        # Phase 1 is the gamma' phase, phase 2 is the gamma phase.
        x_1 = x/sub["nom:1"]["val"]
        x_2 = x_1*sub["1:2"]["val"]
        # Note these are fractional errors
        x_1_stdf = sub["nom:1"]["std"]
        x_2_stdf = np.sqrt(sub["nom:1"]["std"]**2+sub["1:2"]["std"]**2)
        x_1_std  = x_1_stdf*x_1
        x_2_std  = x_2_stdf*x_2
        # Things to calculate in ugly for-loop below.
        x_1_sumto1 = np.empty_like(x_1)
        x_2_sumto1 = np.empty_like(x_2)
        x_1_sumto1_std = np.empty_like(x_1_std)
        x_2_sumto1_std = np.empty_like(x_2_std)
        model_codes = np.empty((N,3),dtype=int)
        f_opt = f_dir.copy()
        f_opt_std = f_dir_std.copy()
        # Yes this SHOULD all be vectorised.
        for a,(x_a,x_1_a,x_2_a,x_1_std_a,x_2_std_a,x_1_stdf_a,x_2_stdf_a) in enumerate(zip(x,x_1,x_2,x_1_std,x_2_std,x_1_stdf,x_2_stdf)):
            best_cal_dist_1 = np.inf
            best_cal_dist_2 = np.inf
            for j,el in enumerate(self.elements):
                if x_a[j] > 0.0:
                    # Calculated elements for each phase
                    x_1_aj = 1. - np.nansum(x_1_a[:j]) - np.nansum(x_1_a[j+1:])
                    x_2_aj = 1. - np.nansum(x_2_a[:j]) - np.nansum(x_2_a[j+1:])
                    if x_1_aj >= 0.0:
                        # Error in the "calculated" element.
                        x_1_std_sq_aj = np.nansum(x_1_std_a[:j]**2) + np.nansum(x_1_std_a[j+1:]**2)
                        # Average error for this model:
                        #avg_x_1_std_aj = np.nansum(x_1_stdf_a[:j]**2) + np.nansum(x_1_stdf_a[j+1:]**2) \
                        #    + x_1_std_sq_aj/x_1_aj**2 
                        cal_dist_1 = (x_1_aj - x_1_a[j])/x_1_std_a[j]
                        # Keep if best uncertainity.
                        if cal_dist_1 < best_cal_dist_1:
                            best_model_1 = j
                            best_cal_dist_1 = cal_dist_1
                            best_comp_1 = x_1_a.copy() ; best_comp_1[j] = x_1_aj
                            best_std_1  = x_1_std_a.copy() ; best_std_1[j] = np.sqrt(x_1_std_sq_aj)
                    if x_2_aj >= 0.0:
                        # Error in the "calculated" element.
                        x_2_std_sq_aj = np.nansum(x_2_std_a[:j]**2) + np.nansum(x_2_std_a[j+1:]**2)
                        # Average error for this model:
                        #avg_x_2_std_aj = np.nansum(x_2_stdf_a[:j]**2) + np.nansum(x_2_stdf_a[j+1:]**2) \
                        #    + x_2_std_sq_aj/x_2_aj**2 
                        cal_dist_2 = (x_2_aj - x_2_a[j])/x_2_std_a[j]
                        if cal_dist_2 < best_cal_dist_2:
                            best_model_2 = j
                            best_cal_dist_2 = cal_dist_2
                            best_comp_2 = x_2_a.copy() ; best_comp_2[j] = x_2_aj
                            best_std_2  = x_2_std_a.copy() ; best_std_2[j] = np.sqrt(x_2_std_sq_aj)
            # Store optimal models found, calculate f
            f_cal = np.nansum((x_a-best_comp_2)*(best_comp_1-best_comp_2))/np.nansum((best_comp_1-best_comp_2)**2)
            f_cal_std = np.sqrt(np.nansum(((2.*f_cal-1.)*best_comp_1+2.*(1.-f_cal)*best_comp_2-x_a)**2*best_std_2**2 +\
                                      (x_a+(2.*f_cal-1.)*best_comp_2-2.*f_cal*best_comp_1)**2*best_std_1**2)) \
                /np.abs(np.nansum((x_a-best_comp_2)*(best_comp_1-best_comp_2))) # fractional error
            if f_cal_std < f_dir_std[a,0]:
                f_opt[a,0] = f_cal
                f_opt_std[a,0] = f_cal_std
                model_codes[a] = np.array([1,best_model_1,best_model_2])
            else:
                model_codes[a] = np.array([0,best_model_1,best_model_2])
            x_1_sumto1[a] = best_comp_1 ; x_1_sumto1_std[a] = best_std_1
            x_2_sumto1[a] = best_comp_2 ; x_2_sumto1_std[a] = best_std_2
        # Return fractional errors in composition to be consistent
        x_1_sumto1_std /= x_1_sumto1
        x_2_sumto1_std /= x_2_sumto1_std
        return x_1_sumto1,x_1_sumto1_std,x_2_sumto1,x_2_sumto1_std,f_opt,f_opt_std,model_codes
            
                    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KERNEL SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
# Sets up kernel using settings in config file
def setup_kernel(kernel_type,feature_range):
    kernel_settings = kernel_type.lower().split("_")
    mixing = False
    if kernel_settings[-1] == "comp":
        comp_style = True
    else:
        comp_style = False
    if kernel_settings[0] == "l1rbf":
        kernel = L1RBF(1.0,(1.e-5,1.e5),dims=n,dim_range=feature_range,comp=comp_style)
    elif kernel_settings[0] == "l2rbf":
        kernel = L2RBF(1.0,(1.e-5,1.e5),dims=n,dim_range=feature_range,comp=comp_style)
    elif kernel_settings[0] == "physrbf":
        kernel = physRBF(1.e4,(1.e-1,1.e8),dims=n,dim_range=feature_range)
    elif kernel_settings[0] == "l1rbfpr":
        kernel = subspace_L1RBF(1.0,(1.e-5,1.e5),groups=groups,num_groups=num_groups,
                                dim_range=feature_range,comp=comp_style)
    elif kernel_settings[0] == "l2rbfpr":
        kernel = subspace_L2RBF(1.0,(1.e-5,1.e5),groups=groups,num_groups=num_groups,
                                dim_range=feature_range,comp=comp_style)
    elif kernel_settings[0] == "l1rbfmixpr":
        init_mix_vals = 0.5*np.ones(sum([1 if np.size(_)>1 else 0 for _  in groups]))
        kernel = subspace_mixing_L1RBF(1.0,(1.e-5,1.e5),
                                       mix_vals=init_mix_vals,
                                       groups=groups,num_groups=num_groups,
                                       dim_range=feature_range,comp=comp_style)
        mixing = True
    elif kernel_settings[0] == "l2rbfmixpr":
        init_mix_vals = 0.5*np.ones(sum([1 if np.size(_)>1 else 0 for _  in groups]))
        kernel = subspace_mixing_L2RBF(1.0,(1.e-5,1.e5),
                                       mix_vals=init_mix_vals,
                                       groups=groups,num_groups=num_groups,
                                       dim_range=feature_range,comp=comp_style)
        mixing = True
    else:
        kernel = None
    return kernel,mixing
kernel,use_mixing = setup_kernel(comp_kernel_0,comp_range)
if kernel:
    for range_,kernel_type in zip((comp_range,ht_range,ht_range),(comp_kernel_1,ht_kernel_0,ht_kernel_1)):
        kernel_1,use_mixing_1 = setup_kernel(kernel_type,range_)
        if kernel_1:
            kernel += ConstantKernel(0.01,(1.e-6,1.)) * kernel_1
        use_mixing = use_mixing or use_mixing_1
else:
    print("Kernel has been misdefined in config file.")
kernel *= ConstantKernel()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA SECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Process database in order to get all the microstructural data.
ms_df = get_microstructure_data(df,drop_duplicate_comps=(not incl_ht),shuffle_seed=seed)
# Split into train and test datasets.
N = ms_df.shape[0]
f_true_kfolds = np.array([])
f_pred_kfolds = np.array([])
f_stds_kfolds = np.array([])
best_models_kfolds = np.array([])
f_all_kfolds = np.empty([n_els+1,0])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% K-FOLDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for fold_i in range(n_folds):
    if n_fold_testing:
        N_test = N//n_folds
        N_train = N - N_test
        train_df = pd.concat([ms_df.iloc[(fold_i+1)*N_test:,:],ms_df.iloc[:fold_i*N_test,:]])
        test_df  = ms_df.iloc[fold_i*N_test:(fold_i+1)*N_test,:]
    else:
        N_train = int(np.rint(N*(1.-test_frac)))
        train_df = ms_df.iloc[:N_train,:]
        test_df  = ms_df.iloc[N_train:,:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODEL FITTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # FITTING PROCEDURE 
    model_k = model(elements,kernel,alpha_noise,seed,
                    standardise_comp,standardise_ht,comp_range,ht_range,
                    kernel_pr_mixing=use_mixing)
    # Get the test/training data
    ml_data_dict = model_k.matching_data_struct()
    ml_data_dict_t = deepcopy(ml_data_dict) # To store the test data.
    for el in elements:
        part_coeff_header = [("γ/γ’ partitioning ratio","at. %",el),("Composition","at. %",el),("γ’ composition","at. %",el)]
        # Get training data:
        x,y = get_Xy(train_df,part_coeff_header,min_max=[0.0,None],ht=incl_ht)
        ml_data_dict[el]["nom:1"] = (x,np.log(y[:,1]/y[:,2]))
        ml_data_dict[el]["1:2"] = (x,np.log(y[:,0]))
        # Get test data
        x,y = get_Xy(test_df,part_coeff_header,min_max=[0.0,None],ht=incl_ht)
        ml_data_dict_t[el]["nom:1"] = (x,y[:,1]/y[:,2])
        ml_data_dict_t[el]["1:2"] = (x,y[:,0])
    # Get training data for f
    x,y = get_Xy(train_df,("γ’ fraction","at. %"),drop_na=False,flatten=True,ht=incl_ht)
    ml_data_dict["f"]["f1"] = (x,np.log(0.01*y))
    # Get test data for f
    x,y = get_Xy(test_df,("γ’ fraction","at. %"),drop_na=False,flatten=True,ht=incl_ht)
    ml_data_dict_t["f"]["f1"] = (x,0.01*y)
    # Test data used to make predictions on/against:
    X_ms_t = x.copy()
    x_prc_target_t = 0.01*(test_df.loc[:,("γ’ composition","at. %")]).drop(["Hf","Nb"],axis=1).astype(np.float64).values
    x_mtx_target_t = 0.01*(test_df.loc[:,("γ composition","at. %")]).drop(["Hf","Nb"],axis=1).astype(np.float64).values
    
    print("\n"+120*"-")
    print(40*"-"+"BEGINNING FIT TO TRAINING DATA FOLD k={:}".format(fold_i)+40*"-")
    print(120*"-"+"\n")
    # Fit model
    model_k.fit_sub_models(ml_data_dict,verbosity=v)
    # Make f predicitions for model.
    f_pred_t,f_std_t,models_t = model_k.predict(X_ms_t)
    
    # Analyse and store results on test data for this fold.
    f_true = ml_data_dict_t["f"]["f1"][1]
    if v>=1:
        print("\nPhase fraction R^2 score for fold {:} = {:5f}".format(fold_i,r2_score(f_true,f_pred_t)))
    # Things we're interested in for all k-folds.
    f_true_kfolds = np.append(f_true_kfolds,f_true)
    f_pred_kfolds = np.append(f_pred_kfolds,f_pred_t)
    best_models_kfolds = np.append(best_models_kfolds,models_t)
    f_stds_kfolds = np.append(f_stds_kfolds,f_std_t)
    
# Some final output
print("\n"+120*"=")
f_comm_true_kfolds = f_true_kfolds[(f_true_kfolds>=0.55) & (f_true_kfolds<=0.85)]
f_comm_pred_kfolds = f_pred_kfolds[(f_true_kfolds>=0.55) & (f_true_kfolds<=0.85)]
print("\nOverall R^2 score on {:}-folds = {:5f}".format(n_folds,r2_score(f_true_kfolds,f_pred_kfolds)))
print("Overall R^2 score for commercial alloys only = {:5f}".format(r2_score(f_comm_true_kfolds,f_comm_pred_kfolds)))
print("Pearson's r for commercial alloys only = {:5f}".format(pearsonr(f_comm_true_kfolds,f_comm_pred_kfolds)[0]))
# Use to get colours for different models for each datapt, e.g. for plotting 
model_colours = np.array(list(map({n:i for i,n in enumerate(set(best_models_kfolds))}.get,best_models_kfolds)))

# Function to plot results
def plot_f_byModel(f_true,f_pred,f_stds,
                   lims=None):
    fig,axs=plt.subplots()
    plt.errorbar(f_true,f_pred,yerr=f_stds,fmt=".",ecolor="k",elinewidth=0.5,zorder=0)
    sc = plt.scatter(f_true,f_pred,marker=".",c=model_colours,cmap="brg",zorder=10)
    if lims==None:
        lims = [min(axs.get_xlim()+axs.get_ylim()),max(axs.get_xlim()+axs.get_ylim())]
    axs.set_xlim(lims)
    axs.set_ylim(lims)
    axs.plot(lims,lims,"--k")
    axs.set_aspect("equal","box")
    axs.set_xlabel("Actual precipitate fraction")
    axs.set_ylabel("Predicted precipitate fraction")
    return fig,axs