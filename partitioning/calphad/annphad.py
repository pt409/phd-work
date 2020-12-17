#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd


# In[13]:


import tensorflow.keras as keras


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[86]:


from sklearn.metrics import r2_score


# <u>Initial models of free energy for two phases in a binary alloy</u><br>Ni--Al binary alloy with at. % Al < 25%.<br> First import free energy data found from DFT.

# In[15]:


df_G0 = pd.read_csv("Ni-Al_fcc.csv",index_col=0)
df_G1 = pd.read_csv("Ni-Al_l12.csv",index_col=0)


# In[277]:


# Fit a simple NN for free energy of each phase.
# Phase 0
def G_init_model(df_G):
    df_G = df_G.sample(frac=1.)
    X = df_G.loc[:,"x1":"T"]
    #X = df_G.loc[:,:"x1"]
    y = df_G.loc[:,"G"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    G = keras.models.Sequential([
        keras.layers.Dense(4,activation="tanh",input_shape=X.shape[1:])
    ])
    for n in range(1):
        G.add(keras.layers.Dense(4,activation="tanh"))
    G.add(keras.layers.Dense(1))
    G.compile(loss="mse",optimizer=keras.optimizers.Adam(lr=0.005))
    history = G.fit(X,y,epochs=200)
    return scaler,G
s0, G0_0 = G_init_model(df_G0)
s1, G1_0 = G_init_model(df_G1)


# In[357]:


# Test these models
dat0 = df_G0.loc[0:2,("x1","G")].to_numpy().T
dat1 = df_G1.loc[0:2,("x1","G")].to_numpy().T
pts0 = np.linspace(dat0[0,0],dat0[0,-1]+0.06,100)
pts1 = np.linspace(dat1[0,1]-0.06,dat1[0,-1],100)
crv0 = G0_0.predict(s0.transform(np.c_[pts0,np.zeros_like(pts0)]))
crv1 = G1_0.predict(s1.transform(np.c_[pts1,np.zeros_like(pts1)]))
#crv0 = G0_0.predict(s0.transform(np.array([pts0]).T))
#crv1 = G1_0.predict(s1.transform(np.array([pts1]).T))
plt.plot(dat0[0],dat0[1],"xr")
plt.plot(dat1[0],dat1[1],"xb")
plt.plot(pts0,crv0,"r")
plt.plot(pts1,crv1,"b")
plt.xlabel("x")
plt.ylabel("G")


# In[281]:


# Test these models
dat0 = df_G0.loc[210:212,("x1","G")].to_numpy().T
dat1 = df_G1.loc[210:212,("x1","G")].to_numpy().T
pts0 = np.linspace(dat0[0,0],dat0[0,-1],100)
pts1 = np.linspace(dat1[0,1],dat1[0,-1],100)
crv0 = G0_0.predict(s0.transform(np.c_[pts0,df_G0.loc[210,"T"]*np.ones_like(pts0)]))
crv1 = G1_0.predict(s1.transform(np.c_[pts1,df_G1.loc[210,"T"]*np.ones_like(pts1)]))
plt.plot(dat0[0],dat0[1],"xr")
plt.plot(dat1[0],dat1[1],"xb")
plt.plot(pts0,crv0,"r")
plt.plot(pts1,crv1,"b")
plt.xlabel("x")
plt.ylabel("G")


# In[284]:


plt.plot(df_G0.loc[:,"T"],df_G0.loc[:,"G"],"x")
plt.plot(df_G0.loc[:,"T"],G0_0.predict(s0.transform(df_G0.loc[:,"x1":"T"])),".")
plt.xlabel("T")
plt.ylabel("G")


# Now try a more traditional fitting method

# In[254]:


# Need a different approach, will have to fit quadratic eq. to data first.
def G0_quad(params,x):
    T = x[-1]
    x = x[:-1]
    G = -params[-2]*T+params[-1]
    for i,xi in enumerate(x):
        G += (params[4*i+0]*T+params[4*i+1])*(xi-params[4*i+2]+params[4*i+3]*T)**2
    return G

def G0_quad_grad(params,x):
    T = x[-1]
    x = x[:-1]
    grads = []
    for i,xi in enumerate(x):
        grads += [T*(xi-params[4*i+2]+params[4*i+3]*T)**2]
        grads += [(xi-params[4*i+2]+params[4*i+3]*T)**2]
        grads += [-2*(params[4*i+0]*T+params[4*i+1])*(xi-params[4*i+2]+params[4*i+3]*T)]
        grads += [2*T*(params[4*i+0]*T+params[4*i+1])*(xi-params[4*i+2]+params[4*i+3]*T)]
    grads += [-T]
    grads += [0]
    return np.array(grads)

#G0_quad_v = np.vectorize(G0_quad)
#G0_quad_grads_v = np.vectorize(G0_quad_grad)

def mse(params,x,G_true):
    mse = 0
    mse_grad = np.zeros_like(params)
    for xj,Gj in zip(x,G_true): 
        mse += 0.5*(G0_quad(params,xj)-Gj)**2
        mse_grad += (G0_quad(params,xj)-Gj)*G0_quad_grad(params,xj)
    return mse,mse_grad    


# In[228]:


from scipy.optimize import minimize


# In[255]:


X = df_G0.loc[:,"x1":"T"].to_numpy()
y = df_G0.loc[:,"G"].to_numpy()
params_init = np.array([1.e-3,5.,0.,0.,1.e-3,0.03])
results = minimize(mse,
                   params_init,
                  args=(X,y),
                  jac=True,
                  method="BFGS")


# In[256]:


results


# In[257]:


# Test these models
dat0 = df_G0.loc[0:2,("x1","G")].to_numpy().T
#dat1 = df_G1.loc[0:2,("x1","G")].to_numpy().T
pts0 = np.linspace(dat0[0,0],dat0[0,-1],100)
#pts1 = np.linspace(dat1[0,1],dat1[0,-1],100)
crv0 = np.array([G0_quad(results.x,x_) for x_ in np.c_[pts0,np.zeros_like(pts0)]])
plt.plot(dat0[0],dat0[1],"xr")
#plt.plot(dat1[0],dat1[1],"xb")
plt.plot(pts0,crv0,"r")
#plt.plot(pts1,crv1,"b")


# <u>ML corrective model</u><br>First need to import some experimental data.

# In[306]:


df_exp = pd.read_csv("Ni-Al_exp.csv")


# In[308]:


df_exp.head()


# In[418]:


X_x = tf.cast(df_exp.loc[:,"x_Al"].values.reshape(-1,1),tf.float32)
X_m = tf.cast(df_exp.loc[:,("x1_Al","f")].values,tf.float32)
X_T = tf.cast(df_exp.loc[:,"T"].values.reshape(-1,1),tf.float32)


# In[629]:


# function to scale tensors
def tensor_scaler_transform(scaler,tensor):
    mean = tf.cast(scaler.mean_,tf.float32)
    std  = tf.cast(scaler.var_**0.5,tf.float32)
    return tf.divide(tf.subtract(tensor,mean),std)


# In[386]:


weights_1 = {
    "h1" : tf.Variable(G0_0.layers[0].get_weights()[0]),
    "h2" : tf.Variable(G0_0.layers[1].get_weights()[0]),
    "out": tf.Variable(G0_0.layers[2].get_weights()[0])
}
biases_1 = {
    "b1" : tf.Variable(G0_0.layers[0].get_weights()[1]),
    "b2": tf.Variable(G0_0.layers[1].get_weights()[1]),
    "out": tf.Variable(G0_0.layers[2].get_weights()[1])
}
weights_2 = {
    "h1" : tf.Variable(G1_0.layers[0].get_weights()[0]),
    "h2" : tf.Variable(G1_0.layers[1].get_weights()[0]),
    "out": tf.Variable(G1_0.layers[2].get_weights()[0])
}
biases_2 = {
    "b1" : tf.Variable(G1_0.layers[0].get_weights()[1]),
    "b2": tf.Variable(G1_0.layers[1].get_weights()[1]),
    # Note final bias doesn't need to be trained i.e. doesn't need to be a variable.
    "out": G1_0.layers[2].get_weights()[1]
}


# In[693]:


weights = [
    tf.Variable(G0_0.layers[0].get_weights()[0]),
    tf.Variable(G0_0.layers[0].get_weights()[1]),
    tf.Variable(G0_0.layers[1].get_weights()[0]),
    tf.Variable(G0_0.layers[1].get_weights()[1]),
    tf.Variable(G0_0.layers[2].get_weights()[0]),
    tf.Variable(G0_0.layers[2].get_weights()[1]),
    # weights for 2nd NN
    tf.Variable(G1_0.layers[0].get_weights()[0]),
    tf.Variable(G1_0.layers[0].get_weights()[1]),
    tf.Variable(G1_0.layers[1].get_weights()[0]),
    tf.Variable(G1_0.layers[1].get_weights()[1]),
    tf.Variable(G1_0.layers[2].get_weights()[0]),
    # Note final bias doesn't need to be trained i.e. doesn't need to be a variable.
    tf.Variable(G1_0.layers[2].get_weights()[1])
]


# In[694]:


@tf.function
def G_nn(X,weights):
    # Replica of free energy NN trained above.
    layer_1 = tf.nn.tanh(
        tf.add(tf.matmul(X,weights[0]),weights[1]))
    layer_2 = tf.nn.tanh(
        tf.add(tf.matmul(layer_1,weights[2]),weights[3]))
    out_ly  = tf.add(tf.matmul(layer_2, weights[4]),weights[5])
    return out_ly

@tf.function
def G_tot_nn(x,X_m,X_T):
    x1 = X_m[:,:1]
    f  = X_m[:,1:]
    X1 = tensor_scaler_transform(s0,tf.concat([x1,X_T],1))
    x2 = tf.divide(tf.subtract(x,tf.multiply(f,x1)),tf.subtract(1.,f))
    X2 = tensor_scaler_transform(s1,tf.concat([x2,X_T],1))
    G_nn_1 = G_nn(X1,weights[:6])
    G_nn_2 = G_nn(X2,weights[6:])
    G_tot = tf.add(tf.multiply(G_nn_1,tf.subtract(1.,f)),tf.multiply(G_nn_2,f))
    G_tot_grad = tf.gradients(G_tot,X_m,stop_gradients=X_m)
    G_tot_hess = tf.einsum("ijkl->ijl",tf.hessians(G_tot,X_m)[0])
    return G_tot, G_tot_grad, G_tot_hess


# In[695]:


@tf.function
def cost_func(G,G_grad,G_hess,num_fts,mu=1.0):
    cost = tf.nn.l2_loss(G_grad)
    # Compute dets of each sub matrix of determinant
    for i in range(num_fts):
        cost += mu*tf.reduce_sum(tf.nn.sigmoid(-tf.linalg.det(G_hess[:,:i+1,:i+1])))
    return cost


# In[696]:


optimiser = tf.optimizers.Adam(learning_rate=5.e-3)
training_epochs = 100

@tf.function
def apply_gradients(optimiser,gradients,weights):
    optimiser.apply_gradients(zip(gradients,weights))


# In[697]:


@tf.function
def train_step(model,x,X_m,X_T):
    with tf.GradientTape() as tape:
        current_cost = cost_func(*model(x,X_m,X_T),2)
    grads = tape.gradient(current_cost,weights)
    #optimiser.apply_gradients(zip(grads,weights_1+weights_2))
    apply_gradients(optimiser,grads,weights)
    print(current_cost)


# In[698]:


# DOESN'T SEEM TO WORK
#for e in range(training_epochs):
#    train_step(G_tot_nn,X_x,X_m,X_T)


# In[699]:


# train NN
for e in range(training_epochs):
    with tf.GradientTape() as tape:
        current_cost = cost_func(*G_tot_nn(X_x,X_m,X_T),2)
    grads = tape.gradient(current_cost,weights)
    apply_gradients(optimiser,grads,weights)
    print("epoch = {:} cost = {:5f}".format(e,current_cost.numpy()))


# In[700]:


weights_0 = [
    tf.Variable(G0_0.layers[0].get_weights()[0]),
    tf.Variable(G0_0.layers[0].get_weights()[1]),
    tf.Variable(G0_0.layers[1].get_weights()[0]),
    tf.Variable(G0_0.layers[1].get_weights()[1]),
    tf.Variable(G0_0.layers[2].get_weights()[0]),
    tf.Variable(G0_0.layers[2].get_weights()[1]),
# weights for 2nd NN
    tf.Variable(G1_0.layers[0].get_weights()[0]),
    tf.Variable(G1_0.layers[0].get_weights()[1]),
    tf.Variable(G1_0.layers[1].get_weights()[0]),
    tf.Variable(G1_0.layers[1].get_weights()[1]),
    tf.Variable(G1_0.layers[2].get_weights()[0]),
    # Note final bias doesn't need to be trained i.e. doesn't need to be a variable.
    tf.Variable(G1_0.layers[2].get_weights()[1])
]


# In[709]:


# Test these models
dat0 = df_G0.loc[150:152,("x1","G")].to_numpy().T
dat1 = df_G1.loc[150:152,("x1","G")].to_numpy().T
pts0 = np.linspace(0.0,0.15,100)
pts1 = np.linspace(0.18,0.3,100)
crv0 = G_nn(tf.cast(s0.transform(np.c_[pts0,df_G0.loc[150,"T"]*np.ones_like(pts0)]),tf.float32),weights_0[:6])
crv1 = G_nn(tf.cast(s1.transform(np.c_[pts1,df_G1.loc[150,"T"]*np.ones_like(pts1)]),tf.float32),weights_0[6:])
crv2 = G_nn(tf.cast(s0.transform(np.c_[pts0,df_G0.loc[150,"T"]*np.ones_like(pts0)]),tf.float32),weights[:6])
crv3 = G_nn(tf.cast(s1.transform(np.c_[pts1,df_G1.loc[150,"T"]*np.ones_like(pts1)]),tf.float32),weights[6:])
plt.plot(dat0[0],dat0[1],"xr")
plt.plot(dat1[0],dat1[1],"xb")
plt.plot(pts0,crv0,"--r")
plt.plot(pts1,crv1,"--b")
plt.plot(pts0,crv2,"r")
plt.plot(pts1,crv3,"b")
plt.xlabel("x")
plt.ylabel("G")


# In[ ]:




