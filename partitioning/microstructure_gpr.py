#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:44:29 2020

@author: Pat Taylor (pt409)
"""

import numpy as np
import pandas as pd
#import pickle

#from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
#from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform

#from itertools import product

#from copy import deepcopy,copy

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
                return K, np.empty(((X_pr).shape[0], (X_pr).shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                length_scale_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                length_scale_gradient = ((X_pr)[:, np.newaxis, :] - (X_pr)[np.newaxis, :, :])**2 \
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
                                *pdist(X_pr,lambda x,y: (x[b]-y[b])/abs(x[b]-y[b])-(x[a]-y[a])/abs(x[a]-y[a])))
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
                return K, np.empty(((X_pr).shape[0], (X_pr).shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                length_scale_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                length_scale_gradient = ((X_pr)[:, np.newaxis, :] - (X_pr)[np.newaxis, :, :])**2 \
                    / length_scale
                length_scale_gradient *= K[..., np.newaxis]
                # Derivative wrt mixing variables
                K_gradient = self.compute_gradient(X,X_pr,K,length_scale_gradient)
                return K, K_gradient
        else:
            return K
        
# Function to print mixing values from a fitted kernel
def get_mixing_vals(elements,gpr_model):
    all_params = gpr_model.kernel_.get_params()
    for param_name,param_value in all_params.items():
        if "mix_vals" in param_name:
            kernel_component = gpr_model.kernel_
            for word in param_name.split("__"):
                if "k1" in word:
                    kernel_component = kernel_component.k1
                elif "k2" in word:
                    kernel_component = kernel_component.k2
            trans = kernel_component.A.T
            for row in trans:
                print(" + ".join([" ".join(_) for _ in np.transpose([list(map("{:.4f}".format,row)),elements])]))
            
      
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA SECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Process database in order to get all the microstructural data.
ms_df = get_microstructure_data(df,drop_duplicate_comps=(not incl_ht),shuffle_seed=seed)
# Split into train and test datasets.
N = ms_df.shape[0]
f_t_kfolds = np.array([])
f_best_kfolds = np.array([])
f_errs_kfolds = np.array([])
best_models_kfolds = np.array([])
f_all_kfolds = np.empty([n_els+1,0])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KERNEL SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
# Sets up kernel using settings in config file
def setup_kernel(kernel_type,feature_range):
    kernel_settings = kernel_type.lower().split("_")
    mixing = False
    if kernel_settings[-1] == "comp":
        comp_style = True
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% K-FOLDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for fold_i in range(n_folds):
    if n_fold_testing:
        N_test = N//n_folds
        N_train = N - N_test
        train_df = pd.concat([ms_df.iloc[(fold_i+1)*N_test:,:],ms_df.iloc[:fold_i*N_test,:]])
        test_df  = ms_df.iloc[fold_i*N_test:(fold_i+1)*N_test,:]
        #opt_models_pkl = ".".join([x+y for x,y in zip(opt_models_pkl_0.split("."),["_{:}".format(fold_i),""])])
        print("---------------------------------------------------\nBEGINNING FIT TO TRAINING DATA SELECTION {:}\n---------------------------------------------------\n".format(fold_i))
    else:
        N_train = int(np.rint(N*(1.-test_frac)))
        train_df = ms_df.iloc[:N_train,:]
        test_df  = ms_df.iloc[N_train:,:]
        #opt_models_pkl = opt_models_pkl_0
    
    output_head = "        \t"+"       \t".join(elements)+"\n"
    
    # Process all the data from a database:
    def process_all_data(df): 
        # Get all the data for precipitate fraction and g/g' partitioning coeff. 
        # Store as tuples in a dict for fast access.
        ml_data_dict = {}
        for el in elements:
            part_coeff_header = [("γ/γ’ partitioning ratio","at. %",el),("Composition","at. %",el),("γ’ composition","at. %",el)]
            if not part_coeff_header[0] in df:
                part_coeff_header = [("γ/γ’ partitioning ratio","at. %",el+" "),("Composition","at. %",el+" "),("γ’ composition","at. %",el+" ")] # In case there's a space after the element name in the database.
            x,y = get_Xy(df,part_coeff_header,min_max=[0.0,None],ht=incl_ht)
            ml_data_dict[el] = (x,np.log(np.c_[y[:,0],y[:,1]/y[:,2]]))
        #ml_data_dict["f"] = get_Xy(df,("γ’ fraction","at. %"),drop_na=False,flatten=True,ht=incl_ht)
        f_data = get_Xy(df,("γ’ fraction","at. %"),drop_na=False,flatten=True,ht=incl_ht)
        # output_head = "        \t"+"       \t".join(elements + ["f"]) # Holdover from when krr model was used for f
        
        # Target data
        #X_ms = ml_data_dict["f"][0]
        X_ms = f_data[0]
        x_comp = 0.01*(df.loc[:,("Composition","at. %")]).drop(["Ni","Hf","Nb"],axis=1).astype(np.float64).values
        x_comp_full = np.c_[1.0-x_comp.sum(axis=1),x_comp]
        x_prc_target = 0.01*(df.loc[:,("γ’ composition","at. %")]).drop(["Hf","Nb"],axis=1).astype(np.float64).values
        #f = ml_data_dict["f"][1].reshape(-1,1)
        f = 0.01*f_data[1].reshape(-1,1)
        return ml_data_dict, f_data, X_ms, x_comp, x_comp_full, x_prc_target, f
    
    # Split into test/training datasets
    ml_data_dict,  f_data,  X_ms,  x_comp,  x_comp_full,  x_prc_target,  f   = process_all_data(train_df)
    ml_data_dict_t,f_data_t,X_ms_t,x_comp_t,x_comp_full_t,x_prc_target_t,f_t = process_all_data(test_df)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODEL FITTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # FITTING PROCEDURE            
    # Stored models...
    models = {}
    scalers = {}
    # Learn partitioning coefficients models for each element and part. coeff.
    for el in elements:
        X = ml_data_dict[el][0]
        if standardise_ht and standardise_comp:
            scaler = PartScaler(with_mean=False)
            X = scaler.fit_transform(X)
        elif standardise_ht:
            scaler = PartScaler(ht_range,with_mean=False)
            X = scaler.fit_transform(X)
        elif standardise_comp:
            scaler = PartScaler(comp_range,with_mean=False)
            X = scaler.fit_transform(X)
        sub_models = {}
        for a,part_coeff_type in enumerate(["1:2","nom:2"]):
            gpr = GaussianProcessRegressor(kernel=kernel,
                                           normalize_y=True,
                                           random_state=seed,
                                           alpha=alpha_noise,
                                           n_restarts_optimizer=2)
            if v>1:
                print("Fitting model for "+el+" phase "+part_coeff_type+" partitioning coefficient...")
            gpr.fit(X,ml_data_dict[el][1][:,a])
            if v>2 and use_mixing:
                print("Pseudo-element representation for model:")
                get_mixing_vals(elements,gpr)
            sub_models[part_coeff_type] = gpr
        models[el] = sub_models
        if standardise_ht or standardise_comp:
            scalers[el] = scaler
    # Repeat for precipitate fraction
    X = f_data[0]
    if standardise_ht and standardise_comp:
        scaler = PartScaler(with_mean=False)
        X = scaler.fit_transform(X)
    elif standardise_ht:
        scaler = PartScaler(ht_range,with_mean=False)
        X = scaler.fit_transform(X)
    elif standardise_comp:
        scaler = PartScaler(comp_range,with_mean=False)
        X = scaler.fit_transform(X)
    f_fit = np.log(0.01*f_data[1])
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   normalize_y=True,
                                   random_state=seed,
                                   alpha=alpha_noise,
                                   n_restarts_optimizer=2)
    if v>1:
        print("Fitting model for phase fraction f.")
    gpr.fit(X,f_fit)
    models["f"] = gpr
    if standardise_ht or standardise_comp:
        scalers["f"] = scaler
    
    # Work out fractions for test data set.
    f_all = []
    f_err_all = []
    # Uncertainties for fractions calculated from part. coeffs.
    for el in elements:
        if standardise_ht or standardise_comp:
            X = scalers[el].transform(X_ms_t)
        else:
            X = X_ms_t.copy()
        K1,K_err1 = models[el]["1:2"].predict(X,return_std=True)
        K2,K_err2 = models[el]["nom:2"].predict(X,return_std=True)
        f_ = (np.exp(K2)-np.exp(K1))/(1-np.exp(K1))
        f_err = ((np.exp(K2)/(1-np.exp(K1)))**2 * K_err1**2 + (np.exp(K1)*(-1+np.exp(K2))/(1-np.exp(K1)**2))**2 * K_err2**2)**0.5
        f_all += [f_]
        f_err_all += [f_err]
    # Uncertainty for fraction calculated directly.
    if standardise_ht or standardise_comp:
        X = scalers["f"].transform(X_ms_t)
    else:
        X = X_ms_t.copy()
    f_d,f_err_d = models["f"].predict(X,return_std=True)
    # Have model of log(f) - convert to f
    f_d = np.exp(f_d)
    f_err_d = f_d*f_err_d
    f_all += [f_d]
    f_err_all += [f_err_d]
    # Find smallest Uncertainty model
    # Exlcude models from elements not present in that particular alloy.
    excl_M1 = np.append(np.where(x_comp_full_t.T==0.0,np.inf,0),np.array([np.zeros(x_comp_full_t.shape[0])]),axis=0)
    excl_M2 = np.append(np.where(x_comp_full_t.T==0.0,np.nan,1),np.array([np.ones(x_comp_full_t.shape[0])]),axis=0)
    f_all = np.array(f_all)*excl_M2
    f_err_all = np.abs(np.array(f_err_all))
    # Locate the best uncertainties to use.
    best_locs = np.where(f_err_all == np.nanmin(f_err_all+excl_M1,axis=0))
    best_locs = (best_locs[0][best_locs[1].argsort()],np.sort(best_locs[1]))
    best_models = best_locs[0] # Store which models were best for which data points on this fold.
    f_best = f_all[best_locs]
    f_best_errs = f_err_all[best_locs] # Want absolute errors here.
    print("\nR^2 score for fold {:} = {:5f}".format(fold_i,r2_score(f_t,f_best)))
    # Things we're interested in for all k-folds.
    f_all_kfolds = np.append(f_all_kfolds,f_all,axis=1)
    f_t_kfolds = np.append(f_t_kfolds,f_t)
    f_best_kfolds = np.append(f_best_kfolds,f_best)
    best_models_kfolds = np.append(best_models_kfolds,best_models)
    f_errs_kfolds = np.append(f_errs_kfolds,f_best_errs)
# Some final output
print("\n\n---------------------------------------------------")
f_comm_t_kfolds = f_t_kfolds[(f_t_kfolds>=0.55) & (f_t_kfolds<=0.85)]
f_comm_best_kfolds = f_best_kfolds[(f_t_kfolds>=0.55) & (f_t_kfolds<=0.85)]
print("\nOverall R^2 score on {:}-folds = {:5f}".format(n_folds,r2_score(f_t_kfolds,f_best_kfolds)))
print("Overall R^2 score for commercial alloys only = {:5f}".format(r2_score(f_comm_t_kfolds,f_comm_best_kfolds)))
print("Pearson's r for commercial alloys only = {:5f}".format(pearsonr(f_comm_t_kfolds,f_comm_best_kfolds)[0]))
# Use to get colours for different models for each datapt, e.g. for plotting 
model_colours = np.array(list(map({n:i for i,n in enumerate(set(best_models_kfolds))}.get,best_models_kfolds)))