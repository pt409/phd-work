#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[2]:


from scipy.optimize import minimize
import scipy.optimize as opt


# <u>Initial models of free energy for two phases in a binary alloy</u><br>Ni--Al binary alloy with at. % Al < 25%.<br> First import free energy data found from DFT.

# In[3]:


df_G1 = pd.read_csv("Ni-Al_fcc.csv",index_col=0)
df_G2 = pd.read_csv("Ni-Al_l12.csv",index_col=0)


# In[4]:


# Shuffle each database then extract features, free energies
def process(df,T_min=400.0):
    df = df[df["T"] >= T_min]
    df  = df.sample(frac=1.)
    X_x = df.loc[:,"x1"]
    X_T = df.loc[:,"T"]
    # Normalise:
    x_mean = X_x.mean()
    x_std = X_x.std()
    T_mean = X_T.mean()
    T_std = X_T.std()
    X_x = (tf.cast(X_x.values.reshape(-1,1),tf.float32)-x_mean)/x_std
    X_T = (tf.cast(X_T.values.reshape(-1,1),tf.float32)-T_mean)/T_std
    y_G = tf.cast(df.loc[:,"G"].values.reshape(-1,1),tf.float32)
    return X_x, X_T, y_G, {"mean":x_mean,"std":x_std}, {"mean":T_mean,"std":T_std}

X_x_1, X_T_1, y_G_1, x_scaling_1, T_scaling_1 = process(df_G1)
X_x_2, X_T_2, y_G_2, x_scaling_2, T_scaling_2 = process(df_G2)


# For the free energies of each phase a partially input convex neural network architecture is used based on <a href="http://proceedings.mlr.press/v70/amos17b/amos17b.pdf">this</a> paper.

# In[5]:


# Regular neural network layer (u)
@tf.function
def u_layer(u_prev,weights):
    return tf.matmul(u_prev,weights[0]) + weights[1]
# Convex neural network layer (z)
@tf.function
def z_layer(u_prev,z_prev,y,weights):
    output = tf.matmul(z_prev * (tf.matmul(u_prev,weights[1]) + weights[2]),weights[0])             + tf.matmul(y * (tf.matmul(u_prev,weights[4]) + weights[5]),weights[3])             + tf.matmul(u_prev,weights[6])             + weights[7]
    return output
# First layer on z path of neural network.
@tf.function
def z1_layer(x,y,weights):
    output = tf.matmul(y * (tf.matmul(x,weights[1]) + weights[2]),weights[0])             + tf.matmul(x,weights[3])             + weights[4]
    return output

# NN for free energy of a single phase. Input convex in X but not T.
@tf.function
def convex_nn(x,y,weights,hidden_layers=2):
    layer_ui = tf.nn.softplus(u_layer(x,weights[0:2]))
    layer_zi = tf.nn.softplus(z1_layer(x,y,weights[2:7]))
    if hidden_layers>1:
        for i in range(1,hidden_layers):
            layer_ui_1= tf.identity(layer_ui)
            layer_ui  = tf.nn.softplus(u_layer(layer_ui_1,weights[10*i-3:10*i-1]))
            layer_zi  = tf.nn.softplus(z_layer(layer_ui_1,layer_zi,y,weights[10*i-1:10*i+7]))
    z_out = z_layer(layer_ui,layer_zi,y,weights[-8:])
    return z_out


# In[6]:


# Input and output shapes for both phases
# Phase 1
n_x_1_input = X_x_1.shape[1]
n_y_1_input = X_T_1.shape[1]
n_z_1_output= y_G_1.shape[1]
# Phase 2
n_x_2_input = X_x_2.shape[1]
n_y_2_input = X_T_2.shape[1]
n_z_2_output= y_G_2.shape[1]
# Setup weights here
widths = [[2,2],[2,2]] # list len must correspond to number of hidden layers...
# ...sublists have len 2; i.e. 2 integers corresponding to width of ith layer on u path and z path respectively.

# Function to generate weights for convex NN (note - seeds not currently implemented)
def convex_weights(n_x_input,n_y_input,n_z_output,widths):
    initialiser = keras.initializers.GlorotNormal()
    nonNeg_init = keras.initializers.RandomUniform(minval=0.2,maxval=0.8)
    # Variables named according to notation in paper cited above.
    # First layer weights and biases
    weights = [[
        [tf.Variable(initialiser([n_x_input,widths[0][0]]),name="~W0"),
         tf.Variable(initialiser([widths[0][0]]),name="~b0")],       
        [tf.Variable(nonNeg_init([n_y_input,widths[0][1]]),
                     constraint=keras.constraints.NonNeg(),name="Wz0"), 
         tf.Variable(nonNeg_init([n_x_input,n_y_input]),
                     constraint=keras.constraints.NonNeg(),name="Wzu0"),    
         tf.Variable(nonNeg_init([n_y_input]),
                     constraint=keras.constraints.NonNeg(),name="bz0"),              
         tf.Variable(initialiser([n_x_input,widths[0][1]]),name="Wu0"), 
         tf.Variable(initialiser([widths[0][1]]),name="b0")]
    ]]
    # Other hidden layer weights and biases
    for i,widths_i in enumerate(widths[1:]):
        i+=1
        weights += [[
        [tf.Variable(initialiser([widths[i-1][0],widths[i][0]]),name="~W{:}".format(i)),    
         tf.Variable(initialiser([widths[i][0]]),name="~b{:}".format(i))],                  
        [tf.Variable(nonNeg_init([widths[i-1][1],widths[i][1]]),
                     constraint=keras.constraints.NonNeg(),name="Wz{:}".format(i)),           
         tf.Variable(nonNeg_init([widths[i-1][0],widths[i-1][1]]),
                     constraint=keras.constraints.NonNeg(),name="Wzu{:}".format(i)),    
         tf.Variable(nonNeg_init([widths[i-1][1]]),
                     constraint=keras.constraints.NonNeg(),name="bz{:}".format(i)),                  
         tf.Variable(initialiser([n_y_input,widths[i][1]]),name="Wy{:}".format(i)),           
         tf.Variable(initialiser([widths[i-1][0],n_y_input]),name="Wyu{:}".format(i)),         
         tf.Variable(initialiser([n_y_input]),name="by{:}".format(i)),                        
         tf.Variable(initialiser([widths[i-1][0],widths[i][1]]),name="Wu{:}".format(i)),
         tf.Variable(initialiser([widths[i][1]]),name="b{:}".format(i))]
    ]]
    # Final layer weights and biases
    i+=1
    weights += [[
        [tf.Variable(nonNeg_init([widths[-1][1],n_z_output]),
                     constraint=keras.constraints.NonNeg(),name="Wz{:}".format(i)),      
         tf.Variable(nonNeg_init([widths[-1][0],widths[-1][1]]),
                     constraint=keras.constraints.NonNeg(),name="Wzu{:}".format(i)),   
         tf.Variable(nonNeg_init([widths[-1][1]]),
                     constraint=keras.constraints.NonNeg(),name="bz{:}".format(i)),            
         tf.Variable(initialiser([n_y_input,n_z_output]),name="Wy{:}".format(i)),        
         tf.Variable(initialiser([widths[-1][0],n_y_input]),name="Wyu{:}".format(i)),    
         tf.Variable(initialiser([n_y_input]),name="by{:}".format(i)),                  
         tf.Variable(initialiser([widths[-1][0],n_z_output]),name="Wu{:}".format(i)),    
         tf.Variable(initialiser([n_z_output]),name="b{:}".format(i))]
    ]]
    return flatten(weights)

# Flatten an arbitrary list
def flatten(list_):
    flatlist = []
    def recurse(list_):
        for i,sublist in enumerate(list_):
            if isinstance(sublist,list):
                recurse(sublist)
            else: 
                flatlist.append(sublist)
    recurse(list_)
    return flatlist


# In[46]:


# Train both neural networks

# Wrapper to let optimiser.apply_gradients work in graph mode.
@tf.function
def apply_gradients(optimiser,gradients,weights):
    optimiser.apply_gradients(zip(gradients,weights))

# Similarly, wrapper for MSE cost function
@tf.function
def MSE(y_true,y_pred):
    return tf.reduce_mean(keras.losses.MSE(y_true,y_pred))
    
# Training step
@tf.function
def train_step(model,optimiser,x,y,weights,z_true):
    with tf.GradientTape() as tape:
        current_cost = MSE(z_true,model(x,y,weights))
        grads = tape.gradient(current_cost,weights)
        apply_gradients(optimiser,grads,weights)
        return current_cost


# In[47]:


# Initial weights
weights_1 = convex_weights(n_x_1_input,n_y_1_input,n_z_1_output,widths)
# Train:
optimiser = tf.optimizers.Adam(0.01)
training_epochs = 1000
# train NN 1
for step in range(training_epochs):
    current_cost = train_step(convex_nn,optimiser,X_T_1,X_x_1,weights_1,y_G_1)
    print("epoch = {:04d} cost = {:6f}".format(step,current_cost.numpy()))


# In[48]:


# For some reason this cell needs to be rerun here...

# Wrapper to let optimiser.apply_gradients work in graph mode.
@tf.function
def apply_gradients(optimiser,gradients,weights):
    optimiser.apply_gradients(zip(gradients,weights))

# Similarly, wrapper for MSE cost function
@tf.function
def MSE(y_true,y_pred):
    return tf.reduce_mean(keras.losses.MSE(y_true,y_pred))
    
# Training step
@tf.function
def train_step(model,optimiser,x,y,weights,z_true):
    with tf.GradientTape() as tape:
        current_cost = MSE(z_true,model(x,y,weights))
        grads = tape.gradient(current_cost,weights)
        apply_gradients(optimiser,grads,weights)
        return current_cost


# In[49]:


# Initial weights
weights_2 = convex_weights(n_x_2_input,n_y_2_input,n_z_2_output,widths)
# Train:
optimiser = tf.optimizers.SGD(0.05)
training_epochs = 1000
# train NN 2
for step in range(training_epochs):
    current_cost = train_step(convex_nn,optimiser,X_T_2,X_x_2,weights_2,y_G_2)
    print("epoch = {:04d} cost = {:6f}".format(step,current_cost.numpy()))


# In[50]:


# Plotting function to plot initial data points and the fitted free energy curves.
# Chooses a random temperature from the database.
def plot_G(df_G1,df_G2,
           G1_model,G2_model,weights_1,weights_2,
           x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
           T_min=400.0):
    T_fixed = df_G1[df_G1["T"] >= T_min].loc[:,"T"].sample(n=1).values[0]
    # Experimental values.
    x1,G1 = df_G1.loc[df_G1["T"]==T_fixed,("x1","G")].values.T
    x2,G2 = df_G2.loc[df_G2["T"]==T_fixed,("x1","G")].values.T
    fig, axs = plt.subplots()
    axs.plot(x1,G1,"xr")
    axs.plot(x2,G2,"xb")
    # Points for plotting free energy curves.
    pts1 = np.linspace(0.,x2.max(),100)
    pts2 = np.linspace(0.,x2.max(),100)
    X_x_1 = tf.cast((pts1.reshape(-1,1)-x_scaling_1["mean"])/x_scaling_1["std"],tf.float32)
    X_x_2 = tf.cast((pts2.reshape(-1,1)-x_scaling_2["mean"])/x_scaling_2["std"],tf.float32)
    X_T_1 = tf.cast((T_fixed*np.ones_like(pts1).reshape(-1,1)-T_scaling_1["mean"])/T_scaling_1["std"],tf.float32)
    X_T_2 = tf.cast((T_fixed*np.ones_like(pts2).reshape(-1,1)-T_scaling_2["mean"])/T_scaling_2["std"],tf.float32)
    G1_crv = G1_model(X_T_1,X_x_1,weights_1).numpy()
    G2_crv = G2_model(X_T_2,X_x_2,weights_2).numpy()
    axs.plot(pts1,G1_crv,"r")
    axs.plot(pts2,G2_crv,"b")
    axs.set_title("T = {:0f}K".format(T_fixed))
    axs.set_xlabel("x")
    axs.set_ylabel("G")
    top_of_plot = max(0.33*(2*min(G1_crv.min(),G2_crv.min())+max(G2_crv.max(),G1_crv.max())),max(G1.max(),G2.max()))
    axs.set_ylim([axs.get_ylim()[0],top_of_plot])
    return fig,axs


# In[55]:


plot_G(df_G1,df_G2,convex_nn,convex_nn,weights_1,weights_2,x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2)


# Load data to use to fit corrective neural network

# In[56]:


# Get experimental phase diagram for Ni-Al
df_pb = pd.read_csv("Ni-Al_pb_Ardell.csv")


# In[57]:


# Fit a curve to this data so that it can be easily sampled.
pb_1 = np.poly1d(np.polyfit(df_pb.loc[:,"T_1"].values,df_pb.loc[:,"x_Al_1"].values,5))
pb_2 = np.poly1d(np.polyfit(df_pb.dropna().loc[:,"T_2"].values,df_pb.dropna().loc[:,"x_Al_2"].values,5))
print(r2_score(pb_1(df_pb.loc[:,"T_1"].values),df_pb.loc[:,"x_Al_1"].values))
print(r2_score(pb_2(df_pb.dropna().loc[:,"T_2"].values),df_pb.dropna().loc[:,"x_Al_2"].values))


# In[58]:


# Sample mixed region of phase diagram for experimental data points
n_exp_pts = 20
T_min = 400.
T_max = 1400.
T_sampled = T_min + (T_max-T_min)*np.random.rand(n_exp_pts)
x1_sampled = pb_1(T_sampled)
x2_sampled = pb_2(T_sampled)
x_sampled = [x1_ + (x2_ - x1_) * np.random.rand() for T_,x1_,x2_ in zip(T_sampled,x1_sampled,x2_sampled)]
f_sampled = (x_sampled-x1_sampled)/(x2_sampled-x1_sampled)
df_exp = pd.DataFrame()
df_exp.insert(0,"T",T_sampled)
df_exp.insert(0,"f",f_sampled)
df_exp.insert(0,"x2_Al",x2_sampled)
df_exp.insert(0,"x1_Al",x1_sampled)
df_exp.insert(0,"x_Al",x_sampled)


# In[59]:


X_x = tf.cast(df_exp.loc[:,"x_Al"].values.reshape(-1,1),tf.float32)
X_m = tf.cast(df_exp.loc[:,("x1_Al","f")].values,tf.float32)
X_T = tf.cast(df_exp.loc[:,"T"].values.reshape(-1,1),tf.float32)


# Can now define and fit total free energy neural network.

# In[60]:


# Weights of this neural network is total of weights for each NN trained above
# Note that the final bias on one of the networks isn't included.
weights = [tf.Variable(wt,constraint=wt.constraint,name=wt.name+"_new") for wt in weights_1 + weights_2[:-1]]
fixed_bias = [tf.identity(weights_2[-1])]


# In[61]:


@tf.function
def G_tot_nn(X_x,X_m,X_T,
             x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
             weights,n_hidden=2):
    # Assuming convex_nn architecture
    n_weights = 7+(n_hidden-1)*10+8
    weights_1 = weights[:n_weights]
    weights_2 = weights[-n_weights+1:] + fixed_bias
    # Divide microstrucutral features into composition and fraction:
    X_1 = X_m[:,:1]
    f  = X_m[:,1:]
    # Temperature musts be scaled differently for each input
    X_T_1 = (X_T - T_scaling_1["mean"])/T_scaling_1["std"]
    X_T_2 = (X_T - T_scaling_2["mean"])/T_scaling_2["std"]
    # Phase 2 composition is calculated from phase 1 composition, f.
    X_2 = ((X_x - (1. - f) * X_1)/f - x_scaling_2["mean"])/x_scaling_2["std"]
    X_1 = (X_1 - x_scaling_1["mean"])/x_scaling_1["std"]
    G_nn_1 = convex_nn(X_T_1,X_1,weights_1)
    G_nn_2 = convex_nn(X_T_2,X_2,weights_2)
    G_tot = G_nn_1 * (1. - f) + G_nn_2 * f
    G_tot_grad = tf.gradients(G_tot,X_m,stop_gradients=X_m)
    G_tot_hess = tf.einsum("ijkl->ijl",tf.hessians(G_tot,X_m)[0])
    return G_tot, G_tot_grad, G_tot_hess


# In[62]:


# Custom cost function to fit minima
@tf.function
def fit_min_cost(G,G_grad,G_hess,num_fts,mu=1.0):
    cost = tf.nn.l2_loss(G_grad)
    # Compute dets of each sub matrix of determinant
    for i in range(num_fts):
        cost += mu*tf.reduce_sum(tf.nn.sigmoid(-tf.linalg.det(G_hess[:,:i+1,:i+1])))
    return cost


# In[63]:


# Can now train neural network
optimiser = tf.optimizers.Adam(5.e-3)
training_epochs = 200
step = 0

# Wrapper to let optimiser.apply_gradients work in graph mode.
@tf.function
def apply_gradients(optimiser,gradients,weights):
    optimiser.apply_gradients(zip(gradients,weights))

@tf.function
def train_step_minfit(model,X_x,X_m,X_T,
             x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
             weights):
    with tf.GradientTape() as tape:
        current_cost = fit_min_cost(*model(X_x,X_m,X_T,
             x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
             weights),2)
        grads = tape.gradient(current_cost,weights)
        apply_gradients(optimiser,grads,weights)
        return current_cost


# In[64]:


while step < training_epochs:
    current_cost = train_step_minfit(G_tot_nn,X_x,X_m,X_T,
             x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
             weights)
    print("epoch = {:04d} cost = {:6f}".format(step,current_cost.numpy()))
    step += 1


# In[67]:


# Plot new free energy curves
plot_G(df_G1,df_G2,convex_nn,convex_nn,weights[:25],weights[-24:]+fixed_bias,x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2)


# Find the phase diagram. Note: G_tot_nn is redefined slightly differently so that it take full input (x1,x2,f). This way constraints can be placed on these inputs during minimisation.

# In[68]:


@tf.function
def G_tot_nn_unconstr(X_m,X_T,
                      x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
                      weights,n_hidden=2):
    # Assuming convex_nn architecture
    n_weights = 7+(n_hidden-1)*10+8
    weights_1 = weights[:n_weights]
    weights_2 = weights[-n_weights+1:] + fixed_bias
    # Divide microstrucutral features into composition and fraction:
    X_1 = X_m[:,0:1]
    X_2 = X_m[:,1:2]
    f   = X_m[:,2:]
    # Temperature musts be scaled differently for each input
    X_T_1 = (X_T - T_scaling_1["mean"])/T_scaling_1["std"]
    X_T_2 = (X_T - T_scaling_2["mean"])/T_scaling_2["std"]
    # Phase 2 composition is calculated from phase 1 composition, f.
    X_2 = (X_2 - x_scaling_2["mean"])/x_scaling_2["std"]
    X_1 = (X_1 - x_scaling_1["mean"])/x_scaling_1["std"]
    G_nn_1 = convex_nn(X_T_1,X_1,weights_1)
    G_nn_2 = convex_nn(X_T_2,X_2,weights_2)
    G_tot = G_nn_1 * (1. - f) + G_nn_2 * f
    G_tot_grad = tf.gradients(G_tot,X_m,stop_gradients=X_m)
    G_tot_hess = tf.einsum("ijkl->ijl",tf.hessians(G_tot,X_m)[0])
    return G_tot, G_tot_grad, G_tot_hess


# In[69]:


# Wrapper to make G_tot_nn_unconstr to work with scipy minimize
def G_tot_2min(x_m,T_fixed,
               x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
               weights,n_hidden=2):
    X_m = tf.cast(x_m.reshape(1,-1),tf.float32)
    G,G_grad,G_hess = G_tot_nn_unconstr(X_m,T_fixed,
                                        x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
                                        weights,n_hidden=2)
    return G.numpy()[0][0],G_grad[0].numpy()[0].astype("float64")


# In[70]:


# Find the phase composition at x,T by minimising total free energy
def minimise_G_tot(x,T,x_m_guess,
                   x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
                   weights):
    x = np.array([x],dtype="float32")
    T_fixed = tf.constant([[T]])
    x_m_guess = np.append(x_m_guess,(x-x_m_guess[0])/(x_m_guess[1]-x_m_guess[0])).astype("float32")
    # Order of variables in x_m is x_1,x_2,f
    # Constraint on microstructure, and explicit functions for Jacobian and Hessians of constraint.
    constr  = lambda x_m : np.array([(1.-x_m[2])*x_m[0]+x_m[2]*x_m[1]])
    constr_jac = lambda x_m : np.array([[1.-x_m[2],x_m[2],-x_m[0]+x_m[1]]])
    constr_hess= lambda x_m,v : np.array([[0.,0.,-v[0]],[0.,0.,v[0]],[-v[0],v[0],0.]])
    results = minimize(G_tot_2min,x_m_guess,
             args=(T_fixed,
                   x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
                   weights),
             method="trust-constr",
             jac=True,
             constraints=opt.NonlinearConstraint(constr,x,x,jac=constr_jac,hess=constr_hess),
             bounds=[(0.0,1.0),(0.0,1.0),(0.0,1.0)])
    return results


# In[71]:


def gen_phase_diagram(x_guess,T_list,
                      x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,weights):
    # x_guess is guess for an element fraction that's in the mixed region at lowest temp in T_list
    T_list = T_list.astype("float32")
    T_list = np.linspace(300.,1600.,20,dtype="float32")
    x_m_guess = np.array([0.05,0.25]) # Guess for solutions to minimisation problem.
    x_m_all = np.array([[]]).reshape(3,0)
    for T_ in T_list:
        result = minimise_G_tot(x_guess,T_,x_m_guess,
                                x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
                                weights)
        x_m_T = result.x
        x_m_all = np.c_[x_m_all,x_m_T.reshape(3,-1)]
        x_guess = 0.5*(x_m_T[0]+x_m_T[1]) # Updated guess for frac in mixed region.
        x_m_guess = x_m_T[:-1] # Updated guess for solutions.
    return x_m_all 
    # Note final row of output is f values: not needed for phase diagram but provide useful check ... 
    # ... that correct solutions (i.e. 0.0 < f < 1.0) were found.


# In[72]:


T_list = np.linspace(300.,1500.,20)
x_pb_old = gen_phase_diagram(0.1,T_list,
                             x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,weights_1+weights_2[:-1])
x_pb_new = gen_phase_diagram(0.15,T_list,
                             x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,weights)


# In[77]:


fig, axs = plt.subplots()
# Real phase diagram, first boundary
axs.plot(pb_1(T_list),T_list,"k",label="Experiment")
# Real phase diagram, second boundary
axs.plot(pb_2(T_list),T_list,"k")
# Old phase diagram, first boundary
axs.plot(x_pb_old[0],T_list,"c",label="Original (DFT)")
# Old phase diagram, second boundary
axs.plot(x_pb_old[1],T_list,"c")
# New phase diagram, first boundary
axs.plot(x_pb_new[0],T_list,"m",label="ML correction")
# New phase diagram, second boundary
plt.plot(x_pb_new[1],T_list,"m")
# Experimental data points used in fit
axs.scatter(df_exp.loc[:,"x_Al"],df_exp.loc[:,"T"],c="k",marker="+",label="Exp. samples")
axs.set_xlabel("x (at. % Al)")
axs.set_ylabel("T (K)")
axs.set_xlim([0.0,0.30])
axs.set_ylim([400.,1500.])
axs.legend()
plt.savefig("phase_diagrams.png",dpi=500)


# In[ ]:





# In[88]:


# Plotting function to plot initial data points and the fitted free energy curves.
# Chooses a random temperature from the database.
def plot_both_G(df_G1,df_G2,
                G1_model,G2_model,
                weights_1i,weights_2i,weights_1c,weights_2c,
                x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2,
                T_min=400.0):
    T_fixed = df_G1[df_G1["T"] >= T_min].loc[:,"T"].sample(n=1).values[0]
    # Experimental values.
    x1,G1 = df_G1.loc[df_G1["T"]==T_fixed,("x1","G")].values.T
    x2,G2 = df_G2.loc[df_G2["T"]==T_fixed,("x1","G")].values.T
    fig, axs = plt.subplots()
    axs.plot(x1,G1,"xc")
    axs.plot(x2,G2,"xc")
    # Points for plotting free energy curves.
    pts1 = np.linspace(0.,x2.max(),100)
    pts2 = np.linspace(0.,x2.max(),100)
    X_x_1 = tf.cast((pts1.reshape(-1,1)-x_scaling_1["mean"])/x_scaling_1["std"],tf.float32)
    X_x_2 = tf.cast((pts2.reshape(-1,1)-x_scaling_2["mean"])/x_scaling_2["std"],tf.float32)
    X_T_1 = tf.cast((T_fixed*np.ones_like(pts1).reshape(-1,1)-T_scaling_1["mean"])/T_scaling_1["std"],tf.float32)
    X_T_2 = tf.cast((T_fixed*np.ones_like(pts2).reshape(-1,1)-T_scaling_2["mean"])/T_scaling_2["std"],tf.float32)
    # Initial models of G
    G1_i_crv = G1_model(X_T_1,X_x_1,weights_1i).numpy()
    G2_i_crv = G2_model(X_T_2,X_x_2,weights_2i).numpy()
    # Corrected models of G
    G1_c_crv = G1_model(X_T_1,X_x_1,weights_1c).numpy()
    G2_c_crv = G2_model(X_T_2,X_x_2,weights_2c).numpy()
    axs.plot(pts1,G1_i_crv,"c")
    axs.plot(pts2,G2_i_crv,"--c")
    axs.plot(pts1,G1_c_crv,"m")
    axs.plot(pts2,G2_c_crv,"--m")
    axs.set_title("T = {:.0f}K".format(T_fixed))
    axs.set_xlabel("x")
    axs.set_ylabel("G")
    top_of_plot = max(0.33*(2*min([G1_i_crv.min(),G2_i_crv.min(),G1_c_crv.min(),G2_c_crv.min()])+max([G2_i_crv.max(),G1_i_crv.max(),G2_c_crv.max(),G1_c_crv.max()])),
                      max(G1.max(),G2.max()))
    axs.set_ylim([axs.get_ylim()[0],top_of_plot])
    return fig,axs


# In[121]:


fig,axs = plot_both_G(df_G1,df_G2,
                      convex_nn,convex_nn,
                      weights_1,weights_2,
                      weights[:25],weights[-24:]+fixed_bias,
                      x_scaling_1,x_scaling_2,T_scaling_1,T_scaling_2)


# In[ ]:




