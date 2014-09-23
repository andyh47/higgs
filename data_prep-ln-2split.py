# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# # Create 2-split training set with log transform on selected vars

# <rawcell>

# test data format:
# EventId,DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt
# 350000,-999.0,79.589,23.916,3.036,-999.0,-999.0,-999.0,0.903,3.036,56.018,1.536,-1.404,-999.0,22.088,-0.54,-0.609,33.93,-0.504,-1.511,48.509,2.022,98.556,0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-0.0

# <codecell>

import platform
if platform.system() == 'Darwin':
    xgboost_path = '/Users/andyh_mac/xgboost/xgboost-master/python'
    number_threads = 8
    data_dir = '/Users/andyh_mac/Desktop/Analytics/Higgs/'
elif platform.system() == 'Linux':
    xgboost_path = '/home/ubuntu/xgboost-master/python'
    number_threads = 32
    data_dir = '/mnt2/Higgs/data_std_3split/'
else:
    print "Don't know parameters for this system: %s" % platform.system()

# <codecell>

import csv
import numpy as np
from matplotlib import pyplot as plt

# <codecell>

# Log transforms for data where range is large
def log_xform(x):
    return np.log(x)
# Check for -999 values
def na(x):
    return np.any(x == -999.0)
# Check for positive values
def pos(x):
    return np.all(x >0)

# <codecell>

# Load training data
print 'Loading training data.'
data_train = np.loadtxt( 'training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
 
# Pick a random seed for reproducible results. Choose wisely!
np.random.seed(42)
# Random number for training/validation splitting
r =np.random.rand(data_train.shape[0])
 
# Put Y(truth), X(data), W(weight), and I(index) into their own arrays
print 'Assigning data to numpy arrays.'
# First 90% are training
Y_train = data_train[:,32][r<0.9]
X_train = data_train[:,1:31][r<0.9]
W_train = data_train[:,31][r<0.9]
# Lirst 10% are validation
Y_valid = data_train[:,32][r>=0.9]
X_valid = data_train[:,1:31][r>=0.9]
W_valid = data_train[:,31][r>=0.9]

X_test =  np.loadtxt( 'test.csv', delimiter=',', skiprows=1 )

# Print dims for inspection
print X_train.shape
print X_valid.shape
print X_test.shape

# <codecell>

# Names of variables to transform
names = 'DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt'
X_colnames = names.split(',')
xform_cols = [2,7,9,10,19,21]
# Check to see that there are no missing values
for i in xform_cols:
    print na(X_train[:,i])
    print na(X_valid[:,i])
    print na(X_test[:,i+1])
# Check to see that all values are positive
for i in xform_cols:
    print pos(X_train[:,i])
    print pos(X_valid[:,i])
    print pos(X_test[:,i+1])

# <codecell>

# Log transform vars
for i in xform_cols:
    X_train[:,i] = log_xform(X_train[:,i])
    X_valid[:,i] = log_xform(X_valid[:,i])
    X_test[:,i+1]= log_xform(X_test[:,i+1])
    plt.figure(i)
    plt.hist(X_test[:,i+1])

# <codecell>

# Write X files
def save_Xfiles(tup):
    print 'Writing file: %s' % tup[0]
    with open(data_dir + tup[0],'w') as fcsv:
        fcsv.write((",").join(tup[1])+'\n')
        for i in range(tup[2].shape[0]):
            data = tup[2][i,:].tolist()
            line = [ str(x) for x in data]
            theline = (",").join(line)+'\n'
            fcsv.write(theline) 
write_list = [ ('X_train_2.csv',X_colnames,X_train), ('X_valid_2.csv',X_colnames,X_valid) ]
for t in write_list:
    save_Xfiles(t)

# <codecell>

# Write Y files
def save_Yfiles(tup):
    print 'Writing file: %s' % tup[0]
    with open(data_dir + tup[0],'w') as fcsv:
        fcsv.write((",").join(tup[1])+'\n')
        for i in range(tup[2].shape[0]):
            data = tup[2][i].tolist()
            line = str(int(data)) + '\n'
            fcsv.write(theline) 
Y_colnames = [ 'Label']
write_list = [ ('Y_train_2.csv',Y_colnames,Y_train), ('Y_valid_2.csv',X_colnames,Y_valid) ]
for t in write_list:
    save_YWfiles(t)

# <codecell>

# Write W files
def save_Wfiles(tup):
    print 'Writing file: %s' % tup[0]
    with open(data_dir + tup[0],'w') as fcsv:
        fcsv.write((",").join(tup[1])+'\n')
        for i in range(tup[2].shape[0]):
            theline = str(tup[2][i]) +'\n'
            fcsv.write(theline) 
W_colnames = [ 'Weight']
write_list = [ ('W_train_2.csv',W_colnames,W_train), ('W_valid_2.csv',W_colnames,W_valid) ]
for t in write_list:
    save_Wfiles(t)

# <codecell>

# Write X test
test_colnames = ['EventId'] + X_colnames
def save_Xfiles(tup):
    print 'Writing file: test_log.csv' 
    with open(tup[0],'w') as fcsv:
        fcsv.write((",").join(tup[1])+'\n')
        for i in range(tup[2].shape[0]):
            data = tup[2][i,:].tolist()
            line = [ str(x) for x in data]
            theline = (",").join(line)+'\n'
            fcsv.write(theline) 
write_list = [ ('test_log.csv',test_colnames,X_test)]
for t in write_list:
    save_Xfiles(t)

