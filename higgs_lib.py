# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import math
# Library of functions used in higgs workbooks

TRAINING_LENGTH = 250000.0

# Calc AMS score
def AMSScore(s,b): 
    return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))

# Count absolute number of signal and background events
def count_s_b(w,y,yhat):
    def getTruePosNeg():
        TruePositive = w*(y==1.0)*(1.0/(w.shape[0]/TRAINING_LENGTH))
        TrueNegative = w*(y==0.0)*(1.0/(w.shape[0]/TRAINING_LENGTH))
        return (TruePositive, TrueNegative)
    pos, neg = getTruePosNeg()
    s = sum( pos*(yhat==1.0) )
    b = sum( neg*(yhat==1.0) )
    return s,b
    
# Calculate the inverse AMS scores and handle exception
def inv_AMSScore(s,b):
    try:
        inv_ams = 1/math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
    except:
        inv_ams= 1
        pass
    return inv_ams

# Make dummy vars for missing jets data and missing der_mass_mmc
# Two new vars:
#      First one is one when DER_MASS_MMC is -999 and zero otherwise
#      Second one is one when PRI_JET_NUM is less than or equal to 1 and zero otherwise
def make_dummies(x):
    DER_MASS_MMC = 0
    PRI_JET_NUM = 22
    missing = np.zeros((x.shape[0],2))
    sel = x[:,DER_MASS_MMC] == -999.0
    missing[sel,0] = 1
    sel = x[:,PRI_JET_NUM] <= 1
    missing[sel,1] = 1
    return missing

