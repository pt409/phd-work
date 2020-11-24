#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:51:34 2020

@author: cdt1801
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import cycler
import numpy as np

# Set color cycle
n = 100
color = plt.cm.winter(np.linspace(0,1,n))
mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color",color)

# Actual vs. predicted plots
def actual_vs_pred_plot(y_true,y_pred,var_name):
    fig,axs=plt.subplots()
    plt.plot(y_true,y_pred,".")
    eqline = [min(axs.get_xlim()+axs.get_ylim()),max(axs.get_xlim()+axs.get_ylim())]
    axs.set_xlim(eqline)
    axs.set_ylim(eqline)
    axs.plot(eqline,eqline,"--k")
    axs.set_aspect("equal","box")
    axs.set_xlabel("Actual "+var_name)
    axs.set_ylabel("Predicted "+var_name)