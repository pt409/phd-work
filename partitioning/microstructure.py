#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:44:29 2020

@author: Pat Taylor (pt409)
"""

import numpy as np
import pandas as pd
import pickle

#from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score # mean_squared_error
from sklearn.model_selection import LeaveOneOut # cross_validate
from sklearn.kernel_ridge import KernelRidge

from scipy.optimize import minimize

from itertools import product

from copy import deepcopy,copy

import configparser
import sys

# Some options.
# Uses a configparser file (.ini structure).
config_default = "rbf_ht_kernel"
config = configparser.ConfigParser()
config.read("microstructure.input")
if len(sys.argv) > 1:
    if sys.argv[1] in config.sections():
        config_type = sys.argv[1]
    else: config_type = config_default
else: config_type = config_default
incl_ht = config[config_type].getboolean("incl_ht")
learn_log_Ki = config[config_type].getboolean("learn_log_Ki")
comp_kernel_type = config[config_type].get("comp_kernel_type")
ht_kernel_type = config[config_type].get("ht_kernel_type")
opt_models_pkl_0 = config[config_type].get("output_models")
database = config[config_type].get("database")
test_frac = config[config_type].getfloat("test_pc")/100.0
seed = config[config_type].getint("seed")
error_weight = config[config_type].getfloat("error_weight")
constr_weight = config[config_type].getfloat("constr_weight")
standardise_ht = config[config_type].getboolean("standardise_ht")
standardise_comp = config[config_type].getboolean("standardise_comp")
prelim_search = config[config_type].getboolean("prelim_search")
gamma_0 = config[config_type].getfloat("gamma")
n_fold_testing = config[config_type].getboolean("n_fold_testing")
n_folds = config[config_type].getint("n_folds")

############################### DATA PROCESSING ###############################

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
    # drop empty rows
    if drop_na:
        sub_df = df.dropna(subset=[y_header])
    else: sub_df = df.copy()
    # drop rows less/greater than certain min/max values
    if min_max:
        min_, max_ = tuple(min_max)
        if min_: sub_df = sub_df[sub_df[y_header] >= min_] # Min
        if max_: sub_df = sub_df[sub_df[y_header] <= max_] # Max
    # Start getting data here:
    y = sub_df[y_header].astype(np.float64).values
    if flatten and len(y.shape) > 1 and y.shape[-1] == 1:
        y = y.flatten()
    if log_y:
        y = np.log(y)
    X1 = 0.01*(sub_df.loc[:,("Composition","at. %")]).drop(drop_els,axis=1).astype(np.float64).values
    if ht:
        X0 = sub_df.loc[:,("Precipitation heat treatment")]
        col_order = sorted(X0.columns.tolist(),key = lambda h: h[1])
        X0 = X0[col_order].replace("-",0.0).astype(np.float64).values
        X0[:,:3] += 273.
        X = np.append(X0,X1,axis=1)
    else:
        X = X1
    return X,y

################################## KERNELS ####################################

# Class to define the unique kernel used.
class rbf_kernel :
    def __init__(self,gamma,dim):
        self.tdim = dim+1
        self.gamma = gamma
        self.Idish = np.r_[np.identity(dim),np.ones((1,dim))]
        self.scale = np.diag(np.ones(self.tdim))
            
    # Calculate kernelised inner product.
    def kernel(self,x,y):
        v = self.scale @ self.Idish @ (x-y)
        return np.exp(-self.gamma*np.inner(v,v))
    
    # Update kernel parameters.
    def update_params(self,gamma):
        self.gamma = gamma
        
    # Add scaling for standardisation
    def add_comp_scaling(self,scale):
        self.scale = np.diag(scale**-1)
        
    @classmethod
    def setup(cls,gamma,dim,scale=None):
        new_instance = cls(gamma,dim)
        if scale:
            new_instance.scale = np.diag(scale**-1)
        return new_instance

class special_kernel(rbf_kernel) :
    def __init__(self,gamma,dim):
        super(special_kernel,self).__init__(gamma,dim)
    
    # Calculate kernelised inner product.
    def kernel(self,x,y):
        return np.exp(-self.gamma*np.linalg.norm(self.scale @ self.Idish @ (x-y),1))

# A special kernel to deal with having composition AND heat treatment.
class multi_kernel(special_kernel):
    def __init__(self,gamma,comp_dim,ht_dim,r=1.):
        # ht_dim is the number of heat treatments (temp+time)
        # r is relative weighting of ht and comp parts of kernel. (NOT USED ANYMORE)
        special_kernel.__init__(self,gamma,comp_dim)
        self.ht_dim = ht_dim
        self.r = r
        self.ht_scale = np.diag(np.ones(2*self.ht_dim))
    
    # Both kernels use l1 norm.
    def kernel(self,x,y):
        x_ht,x0 = self.split_vector(x)
        y_ht,y0 = self.split_vector(y)
        return np.exp(-self.gamma *(np.linalg.norm(self.scale @ self.Idish @ (x0-y0),1)
                                    + np.linalg.norm(self.ht_scale @ (x_ht-y_ht),1)))
    
    def split_vector(self,x):
        x_ht = x[:2*self.ht_dim]
        x0 = x[2*self.ht_dim:]
        return x_ht,x0
    
    # Add scaling for standardisation
    def add_ht_scaling(self,ht_scale):
        self.ht_scale = np.diag(ht_scale**-1)
    
    @classmethod
    def setup(cls,gamma,comp_dim,ht_dim,comp_scale=None,ht_scale=None):
        new_instance = cls(gamma,comp_dim,ht_dim)
        if comp_scale:
            new_instance.scale = np.diag(comp_scale**-1)
        if ht_scale:
            new_instance.ht_scale = np.diag(ht_scale**-1)
        return new_instance
    
# As above but both kernels are simple Gaussian RBF.
class multi_rbf_kernel(multi_kernel):
    def __init__(self,gamma,comp_dim,ht_dim,r=1.):
        super(multi_rbf_kernel,self).__init__(gamma,comp_dim,ht_dim,r)
        
    def kernel(self,x,y):
        x_ht,x0 = self.split_vector(x)
        y_ht,y0 = self.split_vector(y)
        v = self.scale @ self.Idish @ (x0-y0)
        w = self.ht_scale @ (x_ht-y_ht)
        return np.exp(-self.gamma * (np.inner(v,v)
                      + np.inner(w,w)))

### \/ \/ \/ \/ \/ \/ Don't use these (legacy) kernels!!! \/ \/ \/ \/ \/ \/ 
# Variant of above with different kernel for heat treatment
class poly_kernel(multi_kernel):
    def __init__(self,gamma,mu,comp_dim,ht_dim):
        super(poly_kernel,self).__init__(gamma,comp_dim,ht_dim)
        self.mu = mu
    
    # Polynomial kernel
    # ... well, quadratic
    def kernel(self,x,y,offset=1.0):
        x_ht,x0 = self.split_vector(x)
        y_ht,y0 = self.split_vector(y)
        return (np.exp(-self.gamma *(np.linalg.norm(self.scale @ self.Idish@(x0-y0),1)))
                * (self.mu*np.inner(self.ht_scale @ x_ht,self.ht_scale @ y_ht)+offset)**2)
    
    # Update kernel parameters.
    def update_params(self,gamma,mu):
        self.gamma = gamma
        self.mu = mu
        
    @classmethod
    def setup(cls,gamma,mu,comp_dim,ht_dim,comp_scale=None,ht_scale=None):
        new_instance = cls(gamma,mu,comp_dim,ht_dim)
        if comp_scale:
            new_instance.scale = np.diag(comp_scale**-1)
        if ht_scale:
            new_instance.ht_scale = np.diag(ht_scale**-1)
        return new_instance
        
# Another variant (this version uses the original kernel as used in 1st year report).
class unordered_kernel(multi_kernel):
    def __init__(self, gamma, comp_dim, ht_dim,r=1.):
        super(unordered_kernel,self).__init__(gamma,comp_dim,ht_dim,r=r)
        self.ht_scale = np.diag(np.ones(self.ht_dim))
    
    def split_vector(self,x):
        t = x[:self.ht_dim]
        T = x[self.ht_dim:2*self.ht_dim]
        x0 = x[2*self.ht_dim:]
        return t,T,x0
    
    # This kernel separates the heat treatment into "length" and "temperature" parts.
    # Kernel is also independent of order of heat treatments.
    def kernel(self,x,y):
        T0,t0,x0 = self.split_vector(x)
        T1,t1,x1 = self.split_vector(y)
        return np.exp(-self.gamma *(np.linalg.norm(self.Idish@(x0-x1),1)
                      +self.r*np.abs(np.linalg.norm(self.ht_scale@(T0*t0)-np.linalg.norm(self.ht_scale@(T1*t1))))))
    
############################### MAIN SUBROUTINES ##############################
    
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
def predict_phase(models,x_comp,X,
                  elements=["Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"],
                  log_models=False):
    # X is the feature vector (composition or composition & heat treatments)
    # x_comp is the composition of the overall alloy
    N = x_comp.shape[0]
    K = np.empty((N,0))
    f_pred = models["f"].predict(X).reshape(-1,1)
    for el in elements:
        K = np.c_[K,(models[el].predict(X))]
    if log_models:
        K = np.exp(K)
    x_prc = x_comp/((1 - f_pred)*K + f_pred)
    x_prc = np.c_[1.0-x_prc.sum(axis=1),x_prc] # Add on bal element composition
    return x_prc, f_pred

def predict_part_coeffs(models,X,
                        elements=["Ni","Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"],
                        log_models=False):
    N = X.shape[0]
    K = np.empty((N,0))
    for el in elements:
        K = np.c_[K,(models[el].predict(X))]
    if log_models:
        K = np.exp(K)
    return K

def predict_microstructure(models,X_ms,x_comp,x_comp_full):
    k_pred = predict_part_coeffs(models,X_ms,log_models=learn_log_Ki)
    f_pred = np.array([calc_prc_frac(x,k,0.005) for k,x in zip(k_pred,x_comp)]).reshape(-1,1)
    x_prc = x_comp_full/((1.0 - f_pred)*k_pred + f_pred)
    return k_pred, f_pred, x_prc

######################### NUMERICAL POLYNOMIAL SOLVERS #######################
    
# Solve polynomial in f to get the precipitate fraction
def poly_f(part_coeffs,nom_comp):
    k0 = part_coeffs[0]
    k = part_coeffs[1:]
    # The polynomial, term-by-term
    def g(f):
        prc_comp_denom = (1-k)*f+k # A recurring term in the polynomial
        # 1st term
        g = f*(1-k0)*np.prod(prc_comp_denom)
        # 2nd term
        for i, x_i in enumerate(nom_comp):
            g -= (f*(1-k0)+k0)*x_i*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:])
        # 3rd term
        g -= (1-np.sum(nom_comp)-k0)*np.prod(prc_comp_denom)
        return g
    return np.vectorize(g)

# Derivative of the polynomial
def polyd_f(part_coeffs,nom_comp):
    k0 = part_coeffs[0]
    k = part_coeffs[1:]
    # The polynomial's derivative, term-by-term
    def gd(f):
        prc_comp_denom = (1-k)*f+k # A recurring term in the polynomial
        # 1st term, 1st part from product rule
        gd = (1-k0)*np.prod(prc_comp_denom)
        # 1st term, 2nd part from product rule
        for i,ki in enumerate(k):
            gd += f*(1-k0)*(1-ki)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:])
        # 2nd term, 1st part from product rule
        for i, x_i in enumerate(nom_comp):
            gd -= (1-k0)*x_i*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:])
        # 2nd term, 2nd part from prodcut rule
        for i, x_i in enumerate(nom_comp):
            for l,kl in enumerate(k):
                if l!=i:
                    if l<i:
                        gd -= (f*(1-k0)+k0)*x_i*(1-kl)*np.prod(prc_comp_denom[:l])*np.prod(prc_comp_denom[l+1:i])*np.prod(prc_comp_denom[i+1:])
                    elif i<l:
                        gd -= (f*(1-k0)+k0)*x_i*(1-kl)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:l])*np.prod(prc_comp_denom[l+1:])
        # 3rd term
        for i,ki in enumerate(k):
            gd -= (1-np.sum(nom_comp)-k0)*(1-ki)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:])
        return gd
    return np.vectorize(gd)

# Calculate $\prod_{i,i\neq\{j,k,l,...\}}x_i$
def skip_prod(array,indices):
    # indices should be list.
    # Calculates product of array excl. indices supplied.
    indices.sort()
    prod = np.prod(array[:indices[0]])
    if len(indices)>1:
        for pos,index in enumerate(indices[:-1]):
            next_ind = indices[pos+1]
            prod *= np.prod(array[index+1:next_ind])
    prod *= np.prod(array[indices[-1]+1:])
    return prod

# 2nd Derivative of the polynomial
def polydd_f(part_coeffs,nom_comp):
    k0 = part_coeffs[0]
    k = part_coeffs[1:]
    # The polynomial's 2nd derivative, term-by-term
    def gdd(f):
        prc_comp_denom = (1-k)*f+k # A recurring term in the polynomial
        gdd = 0
        # 1st term, 1st 2 parts from product rule
        for i, ki in enumerate(k):
            gdd += 2*(1-k0)*(1-ki)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:])
        # 1st term 3rd part from product rule
        for i, ki in enumerate(k):
            for l,kl in enumerate(k):
                if l!=i:
                    if l<i:
                        gdd += f*(1-k0)*(1-ki)*(1-kl)*np.prod(prc_comp_denom[:l])*np.prod(prc_comp_denom[l+1:i])*np.prod(prc_comp_denom[i+1:])
                    elif i<l:
                        gdd += f*(1-k0)*(1-ki)*(1-kl)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:l])*np.prod(prc_comp_denom[l+1:])
        # 2nd term, 1st 2 parts from product rule
        for i, x_i in enumerate(nom_comp):
            for l,kl in enumerate(k):
                if l!=i:
                    if l<i:
                        gdd -= 2*(1-k0)*x_i*(1-kl)*np.prod(prc_comp_denom[:l])*np.prod(prc_comp_denom[l+1:i])*np.prod(prc_comp_denom[i+1:])
                    elif i<l:
                        gdd -= 2*(1-k0)*x_i*(1-kl)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:l])*np.prod(prc_comp_denom[l+1:])
        # 2nd term, 3rd part from prodcut rule
        for i,x_i in enumerate(nom_comp):
            for l,kl in enumerate(k):
                if l!=i:
                    for m,km in enumerate(k):
                        if m!=i and m!=l:
                            gdd -= (f*(1-k0)+k0)*x_i*(1-kl)*(1-km)*skip_prod(prc_comp_denom, [i,l,m])
        # 3rd term
        for i, k_i in enumerate(k):
            for l,kl in enumerate(k):
                if l!=i:
                    if l<i:
                        gdd -= (1-np.sum(nom_comp)-k0)*(1-k_i)*(1-kl)*np.prod(prc_comp_denom[:l])*np.prod(prc_comp_denom[l+1:i])*np.prod(prc_comp_denom[i+1:])
                    elif i<l:
                        gdd -= (1-np.sum(nom_comp)-k0)*(1-k_i)*(1-kl)*np.prod(prc_comp_denom[:i])*np.prod(prc_comp_denom[i+1:l])*np.prod(prc_comp_denom[l+1:])
        return gdd
    return np.vectorize(gdd)

# Solvers themselves, only halley used, rest are legacy.

def newtonRaphson(g,g_deriv,f0,f_tol):
    df = np.inf
    f = f0
    while abs(df) > f_tol:
        df = g(f)/g_deriv(f)
        f -= df
    return f

def halley(g,g_deriv,g_deriv2,f0,f_tol,N_iter=100):
    df = np.inf
    f = f0
    for n in range(N_iter):
        df = 2*g(f)*g_deriv(f)/(2*g_deriv(f)**2-g(f)*g_deriv2(f))
        f -= df
        if abs(df) < f_tol: break
    return f

def newtonRaphson_bounded(g,g_deriv,f0,f_tol,alpha=1.0,x0=0.5):
    # estimate the tolerance in the function g
    g_tol = f_tol*(np.array([g_deriv(f_) for f_ in np.linspace(0,1,25)]).max())
    # mapping function
    sigma = lambda x: (1+np.exp(-alpha*(x-x0)))**-1
    f = f0
    x = x0 - np.log(-1+f0**-1)/alpha
    g_err = g(sigma(x))
    while abs(g_err) > g_tol:
        g_err = g(sigma(x))
        dx = -4*alpha*(np.cosh(alpha*x/2))**2*g_err/g_deriv(sigma(x))
        x -= dx
    f = sigma(x)
    return f

# More useful wrappers for solvers.

def root_finder(g,g_deriv,g_deriv2,f_tol,f_scale,N_iter=100):
    # Coarse method to find 1st turning point before f=1.0
    f = 1.0
    gd_0 = g_deriv(f)
    while f>0.0:
        f -= f_scale
        gd_1 = g_deriv(f)
        if gd_0*gd_1 <= 0.0: # check for sign change
            gdd_0 = g_deriv2(f+f_scale)
            break
        gd_0 = copy(gd_1)
    # Now search for change in 2nd derivative too
    while f>0.0:
        gdd_1 = g_deriv2(f)
        if gdd_0*gdd_1 <= 0.0:
            break
        f -= f_scale
        gdd_0 = copy(gdd_1)
    f_sol = halley(g,g_deriv,g_deriv2,f,f_tol,N_iter=N_iter)
    return f_sol

def calc_prc_frac(x_comp,part_coeffs,f_tol,
                  f_scale=0.005,trace_tol=0.002,N_iter=100,
                  robust=True):
    # Neglect elements with fraction less than trace_tol
    # f_scale, N_iter options get passed to root_finder
    # robust: using this option, f is also calculated from the inverse partitioning coefficients
    #         to ensure that the correct value has been found.
    trace_els = np.where(x_comp<=trace_tol)[0]
    x = np.delete(x_comp,trace_els)
    k = np.delete(part_coeffs,trace_els+1)    
    p = poly_f(k,x) ; pd = polyd_f(k,x) ; pdd = polydd_f(k,x)
    f = root_finder(p,pd,pdd,f_tol,f_scale)
    if robust:
        # Find root using inverse part. coeffs too
        q = poly_f(k**-1,x) ; qd = polyd_f(k**-1,x) ; qdd = polydd_f(k**-1,x)
        f_ = 1 - root_finder(q,qd,qdd,f_tol,f_scale)
        if abs(f-f_)>2*f_tol:
            if abs(p(f_))<abs(q(f)):
                f = f_
    return f
    
############################### DATA SECTION ##################################

# Process database in order to get all the microstructural data.
ms_df = get_microstructure_data(df,drop_duplicate_comps=(not incl_ht),shuffle_seed=seed)
# Split into train and test datasets.
N = ms_df.shape[0]
for fold_i in range(n_folds):
    if n_fold_testing:
        N_test = N//n_folds
        N_train = N - N_test
        train_df = pd.concat([ms_df.iloc[(fold_i+1)*N_test:,:],ms_df.iloc[:fold_i*N_test,:]])
        test_df  = ms_df.iloc[fold_i*N_test:(fold_i+1)*N_test,:]
        opt_models_pkl = ".".join([x+y for x,y in zip(opt_models_pkl_0.split("."),["_{:}".format(fold_i),""])])
        print("---------------------------------------------------\nBEGINNING FIT TO TRAINING DATA SELECTION {:}\n---------------------------------------------------\n".format(fold_i))
    else:
        N_train = int(np.rint(N*(1.-test_frac)))
        train_df = ms_df.iloc[:N_train,:]
        test_df  = ms_df.iloc[N_train:,:]
        opt_models_pkl = opt_models_pkl_0
    
    elements = ["Ni","Cr","Co","Re","Ru","Al","Ta","W","Ti","Mo"]
    output_head = "        \t"+"       \t".join(elements)+"\n"
    
    # Process all the data from a database:
    def process_all_data(df): 
        # Get all the data for precipitate fraction and g/g' partitioning coeff. 
        # Store as tuples in a dict for fast access.
        ml_data_dict = {}
        for el in elements:
            part_coeff_header = ("γ/γ’ partitioning ratio","at. %",el)
            if not part_coeff_header in df:
                part_coeff_header = ("γ/γ’ partitioning ratio","at. %",el+" ") # In case there's a space after the element name in the database.
            ml_data_dict[el] = get_Xy(df,part_coeff_header,min_max=[0.0,100.0],ht=incl_ht,log_y=learn_log_Ki)
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
    
    ################################# SETUP KERNEL ################################
    
    # Setup initial kernel.
    comp_dim = x_comp.shape[1]
    if incl_ht:
        ht_dim = (X_ms.shape[1]-comp_dim)//2
        # Have 3 different options for heat treatment kernels
        if ht_kernel_type == "poly":
            my_kernel = poly_kernel.setup(0.1,0.1,comp_dim,ht_dim)
            if standardise_ht:
                s = 4*np.std(X_ms[:,:2*ht_dim],axis=0)
                my_kernel.add_ht_scaling(s)
        elif ht_kernel_type == "rbf":
            if comp_kernel_type == "rbf":
                my_kernel = multi_rbf_kernel.setup(0.1,comp_dim,ht_dim)
            else:
                my_kernel = multi_kernel.setup(0.1,comp_dim,ht_dim)
            if standardise_ht:
                s = 4*np.std(X_ms[:,:2*ht_dim],axis=0)
                my_kernel.add_ht_scaling(s)
        else:
            my_kernel = unordered_kernel.setup(0.1,comp_dim,ht_dim)
            if standardise_ht:
                s = 4*np.std(X_ms[:,:ht_dim]*X_ms[:,ht_dim:2*ht_dim],axis=0)
                my_kernel.add_ht_scaling(s)
        # Add scaling for composition
        if standardise_comp:
            s = 4*np.std(np.r_[np.identity(comp_dim),np.ones((1,comp_dim))] @ X_ms[:,-comp_dim:].T,axis=1)
            my_kernel.add_comp_scaling(s)
        
    else:
        # Learn part. coeffs. from composition only.
        # Have 2 different options for kernels.
        if comp_kernel_type == "special":
            my_kernel = special_kernel.setup(0.1,comp_dim) # 9 is dimensionality of composition data.
        else: 
            my_kernel = rbf_kernel.setup(0.1,comp_dim)
        # Add scaling as needed.
        if standardise_comp:
            s = 4*np.std(np.r_[np.identity(comp_dim),np.ones((1,comp_dim))] @ X_ms.T,axis=1)
            my_kernel.add_comp_scaling(s)
            
    ################################ KRR SUBROUTINE ###############################
    
    # inner-most function of the main procedure, trains a krr model for a given property...
    # ... for an entire cohort so that this can be used in a minimization function to optimise...
    # ... cross-validation score.
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
    
    ###############################################################################
    
    # Initial error calculations:
    # can calculate an error if mean compositions were used here.
    mu  = 1.*error_weight # Weighting of precipitate fraction error in the overall error.
    mu2 = 1.*constr_weight # Weighting of soft constraint on part coeff scores in the error.
    frac_error_0 = 0.5*((f - f.mean())**2).mean(axis=0)
    phase_error_0 = (((f*x_prc_target - f.mean()*x_prc_target.mean(axis=0))**2).sum(axis=1,keepdims=True)).mean(axis=0)
    error_0 = mu*frac_error_0 + phase_error_0
    # And repeat for test dataset
    frac_error_0_t = 0.5*((f_t - f_t.mean())**2).mean(axis=0)
    phase_error_0_t = (((f_t*x_prc_target_t - f_t.mean()*x_prc_target_t.mean(axis=0))**2).sum(axis=1,keepdims=True)).mean(axis=0)
    error_0_t = mu*frac_error_0_t + phase_error_0_t
    
    # Kernel ridge model for each microstructural property.
    next_alpha = {ms_prop:0.1 for ms_prop in ml_data_dict.keys()}
    models = {}
    opt_models = {}
    best_error = np.inf
    # Define as a function for use with minimiser. This is used to optimise the kernel.
    def calc_microstruc_error(kernel_params,v=1):
        # v is the verbosity, 0, 1 or 2 (most verbose).
        # First update the kernel with new parameters:
        global my_kernel
        my_kernel.update_params(*kernel_params)    
        if v >= 1:
            print("Kernel parameters for this iteration:")
            print("gamma =\t"+("\t".join("{:.6e}".format(_) for _ in kernel_params)))
            print("\nBeginning to fit kernel ridge models for partitioning coefficients.")
        # Loop through each microstructural property and train optimal krr model using cv.
        # Store some values calculated for output purposes.
        lambda_output = "lambda =\t"
        score_output =  "R^2    =\t"
        cv_output =  "CV err =\t"
        # A term that involves the sign of the R^2 scores and is added to the overall error as a soft constraint
        err_soft_cnstr = 0
        for ms_prop, ms_data in ml_data_dict.items():
            if v>=2: print("{} part. coeff. ...".format(ms_prop))
            result = minimize(train_cohort_model,
                              next_alpha[ms_prop],
                              args=ms_data,
                              method="L-BFGS-B",
                              bounds=[(1.e-8,None)],
                              options={"ftol":1.e-3,
                                       "gtol":1.e-3,
                                       "eps":0.005})
            opt_alpha = result.x
            #next_alpha[ms_prop] = opt_alpha
            krr_model = train_cohort_model(opt_alpha,*ms_data,return_model=True)
            models[ms_prop] = krr_model
            part_coeff_score = krr_model.score(*ms_data)
            err_soft_cnstr += np.heaviside(-part_coeff_score,1.0)
            if v >= 2:
                lambda_output += "{:.5e}\t".format(result.x[0])
                try: 
                    cv_output += "{:.5e}\t".format(result.fun)
                except TypeError:
                    return result
                score_output += "{:.5f}   \t".format(part_coeff_score)
        if v >= 1: print("Done!\n")
        if v >= 2:
            print(output_head)
            print(score_output)
            print(cv_output)
            print(lambda_output)
        # Calculate the predicted phase composition. 
        #x_prc, f_pred = predict_phase(models,x_comp,X_ms,log_models=learn_log_Ki)
        #frac_error = (0.5*(f - f_pred)**2).mean(axis=0)
        #phase_error = ((1.e-4*(f_pred*x_prc - f*x_prc_target)**2).sum(axis=1,keepdims=True)).mean(axis=0)
        #error = mu*frac_error + phase_error
        # New error calculation based on the predicted precipitate fraction
        k_pred = predict_part_coeffs(models,X_ms,log_models=learn_log_Ki)
        #f_pred = np.array([root_finder(poly_f(k,x),polyd_f(k,x),polydd_f(k,x),0.005,0.1) for k,x in zip(k_pred,x_comp)]).reshape(-1,1)
        f_pred = np.array([calc_prc_frac(x,k,0.005) for k,x in zip(k_pred,x_comp)]).reshape(-1,1)
        x_prc = x_comp_full/((1.0 - f_pred)*k_pred + f_pred)
        frac_error = 0.5*((f - f_pred)**2).mean(axis=0)
        # old version:
        #phase_error = (((f_pred*x_prc - f*x_prc_target)**2).sum(axis=1,keepdims=True)).mean(axis=0)
        # new version:
        phase_error = (((f_pred*x_prc - f*x_prc_target)**2).sum(axis=1,keepdims=True)).mean(axis=0)    
        error = mu*frac_error + phase_error + mu2*err_soft_cnstr
        score = 1.0 - error/error_0
        if v >= 2:
            frac_score = 1.0 - frac_error/frac_error_0
            phase_score = 1.0 - phase_error/phase_error_0
            print("\nFraction error = {:.6f}   |   score = {:.5f}".format(frac_error[0],frac_score[0]))
            print("Phase error    = {:.6f}   |   score = {:.5f}".format(phase_error[0],phase_score[0]))
            #print("\nMicrostructural error = {:.6f}\nscore = {:.5f}\n".format(error[0],score[0]))
        # Store models if the error is lowest yet.
        global best_error
        global opt_models
        if error < best_error:
            opt_models = deepcopy(models)
            best_error = copy(error)
            if v >= 1:
                print("Overall error  = {:.6f}   |   score = {:.5f}  <----NEW BEST\n".format(error[0],score[0]))
                print("".join(130*["-"])+"\n")
        elif v >= 1:
            print("Overall error  = {:.6f}   |   score = {:.5f}\n".format(error[0],score[0]))
            print("".join(130*["-"])+"\n")
        # Finally return error
        return error
    
################################ FINAL LOOP ###################################
    
    if __name__ == '__main__':
        print("\n\n+++++++++++++++++++ PARAMETERS +++++++++++++++++++\n\n")  
        [print("{:}\t: {:}".format(key,value)) for key,value in dict(config[config_type]).items()]
        print("\n\n+++++++++++++++++ MAIN ROUTINE +++++++++++++++++\n\n")    
        # Now minimise the microstructural error over the kernel parameters.
        v = 2 # verbosity of output
        if ht_kernel_type == "poly":
            kernel_params_init = np.array([gamma_0,1.e-3])
        else:
            kernel_params_init = np.array([gamma_0])
        # Carry out a preliminary search of the parameters. 
        if prelim_search:
            print("\nStarting initial grid search for kernel parameters...\n")
            for kernel_params in product(*list(np.array([(10**m)*kernel_params_init for m in range(-1,2)]).T)):
                kernel_params = np.array(kernel_params)
                new_error = calc_microstruc_error(kernel_params,v=v)
                if new_error == best_error:
                    kernel_params_init = copy(kernel_params)
            print("\n---------------------------------------------------\nGrid search complete.\n---------------------------------------------------\n")
        eps = 1.e-1*kernel_params_init
        bounds = kernel_params_init.shape[0]*[(0.,None)]
        print("\nStarting LBFGS optimisation of kernel parameters...\n")
        result = minimize(calc_microstruc_error,
                          kernel_params_init,
                          args=(v,),
                          method="L-BFGS-B",
                          bounds=bounds,
                          options={"ftol":1.e-3,
                                   "gtol":1.e-3,
                                   "eps":eps})
        # Pickle the optimised models that were found.
        print("\n\n++++++++++++++ OPTIMISATION COMPLETE ++++++++++++++")  
        with open(opt_models_pkl,"wb") as pickle_out:
            pickle.dump(opt_models,pickle_out)
        print("\nResult of fitting kernel parameters:\n")
        print(result)
        # Print results of model on test data
        print("\n\n++++++++++++++ TEST DATA STATISTICS ++++++++++++++")
        print("\nNumber of microstructures in training data set = {:}\n".format(N_train))
        print("Number of microstructures in test data set     = {:}\n".format(N-N_train))
        # Calculations for test dataset
        k_pred_t = predict_part_coeffs(opt_models,X_ms_t,log_models=learn_log_Ki)
        f_pred_t = np.array([calc_prc_frac(x,k,0.005) for k,x in zip(k_pred_t,x_comp_t)]).reshape(-1,1)
        x_prc_t = x_comp_full_t/((1.0 - f_pred_t)*k_pred_t + f_pred_t)
        frac_error = 0.5*((f_t - f_pred_t)**2).mean(axis=0)
        # old version:
        #phase_error = (((f_pred*x_prc - f*x_prc_target)**2).sum(axis=1,keepdims=True)).mean(axis=0)
        # new version:
        phase_error = (((f_pred_t*x_prc_t - f_t*x_prc_target_t)**2).sum(axis=1,keepdims=True)).mean(axis=0)    
        error = mu*frac_error + phase_error
        score = 1.0 - error/error_0_t
        frac_score = 1.0 - frac_error/frac_error_0_t
        phase_score = 1.0 - phase_error/phase_error_0_t
        print("\nFraction error = {:.6f}   |   score = {:.5f}".format(frac_error[0],frac_score[0]))
        print("Phase error    = {:.6f}   |   score = {:.5f}".format(phase_error[0],phase_score[0]))
        print("Overall error  = {:.6f}   |   score = {:.5f}\n".format(error[0],score[0]))