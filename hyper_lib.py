# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Functions for ploting and selecting hyperOpt parameters
import matplotlib.pyplot as plt
import numpy as np
import pickle
"""
functions: plot_trials(trials), which_tid(n,trials)
"""

def plot_trials(trials):
    def gen_vars():
        result = []
        for i,t in enumerate(trials):
            result.append( (t['misc']['tid'],t['result']['loss'], t['result']['valid_loss']) )
        return result
    v = gen_vars()
    sorted_v = sorted(v, key=lambda x: x[1], reverse=True)
    loss = [ x[1] for x in sorted_v]
    valid_loss = [ x[2] for x in sorted_v]
    label = [ x[0] for x in sorted_v]
    fig =plt.figure(3)
    plt.plot(loss,'o',valid_loss,'^')
    #plt.plot(valid_loss,'^')
    ax.legend(['Training', 'Validation'])
    plt.title('Training Loss, Validation Loss')
    plt.xlabel('Trial')
    plt.ylabel('Loss (1/AMS)')
    plt.show()
    
def which_tid(n,trials):
    def gen_vars():
        result = []
        for i,t in enumerate(trials):
            result.append( (t['misc']['tid'],t['result']['loss'], t['result']['valid_loss']) )
        return result
    v = gen_vars()
    sorted_v = sorted(v, key=lambda x: x[1], reverse=True)
    label = [ x[0] for x in sorted_v]
    idx = label[n]
    print [ x for i,x in enumerate(trials) if i== idx]
    return

    

