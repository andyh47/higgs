# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Use Hyperopt to identify best xgboost parameters 

# <rawcell>

# 
# test data format:
# EventId,DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt
# 350000,-999.0,79.589,23.916,3.036,-999.0,-999.0,-999.0,0.903,3.036,56.018,1.536,-1.404,-999.0,22.088,-0.54,-0.609,33.93,-0.504,-1.511,48.509,2.022,98.556,0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-0.0

# <codecell>

import platform
if platform.system() == 'Darwin':
    xgboost_path = '/Users/andyh_mac/xgboost/xgboost-master/python'
    number_threads = 8
    data_dir = '/Users/andyh_mac/Desktop/Analytics/Higgs/data_std_2split/'
elif platform.system() == 'Linux':
    xgboost_path = '/home/ubuntu/xgboost-master/python'
    number_threads = 32
    data_dir = '/mnt2/Higgs/data_std_3split/'
else:
    print "Don't know parameters for this system: %s" % platform.system()

# <codecell>

from hyperopt import fmin, rand, tpe, space_eval, hp, Trials
from hyperopt import STATUS_OK, STATUS_FAIL
from matplotlib import pyplot as plt
import higgs_lib
import hyper_lib
import math
import numpy as np
import pickle
from pymongo import MongoClient
import sys
sys.path.append(xgboost_path)
import xgboost as xgb

# <codecell>

# Pick a random seed for reproducible results. Choose wisely!
np.random.seed(42)

# Put Y(truth), X(data), W(weight), and I(index) into their own arrays
print 'Assigning data to numpy arrays.'
# First 80% are training
Y_train = np.loadtxt( data_dir + 'Y_train_2.csv', delimiter=',', skiprows=1 )
X_train = np.loadtxt( data_dir + 'X_train_2.csv', delimiter=',', skiprows=1 )
W_train = np.loadtxt( data_dir + 'W_train_2.csv', delimiter=',', skiprows=1 )
# Next 10% are validation
Y_valid = np.loadtxt( data_dir + 'Y_valid_2.csv', delimiter=',', skiprows=1 )
X_valid = np.loadtxt( data_dir + 'X_valid_2.csv', delimiter=',', skiprows=1 )
W_valid = np.loadtxt( data_dir + 'W_valid_2.csv', delimiter=',', skiprows=1 )
weight = W_train * (float(X_train.shape[0]) / len(Y_train))

# Compute weight characteristics
sum_wpos = sum( weight[i] for i in range(len(Y_train)) if Y_train[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(Y_train)) if Y_train[i] == 0.0  )

# <codecell>

# Optional - Make dummy vars for missing jets data and missing der_mass_mmc
# Two new vars:
# First one is one when DER_MASS_MMC is -999 and zero otherwise
# Second one is one when PRI_JET_NUM is less than or equal to 1 and zero otherwise
USE_DUMMIES = True
if USE_DUMMIES == True:
    X_train = np.hstack((X_train, higgs_lib.make_dummies(X_train)))
    X_valid = np.hstack((X_valid, higgs_lib.make_dummies(X_valid)))

# Format data for xgboost
dtrain = xgb.DMatrix(X_train,label=Y_train, weight = weight)
dvalid = xgb.DMatrix(X_valid,label=Y_valid)

# <codecell>

# Set up hyperopt parameter generation
space =  [  hp.uniform('depth',3,8), hp.uniform('eta',.05,.5),hp.quniform('threshold',60,90,1)]

# Define hyperopt objective function optimizing for AMS
n_boost_iter=100  # n_boost_iters for each hyper param iter
def objective(args):
    depth,eta,threshold = args
    threshold = int(threshold)
    depth = int(depth)
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logitraw'
    # scale weight of positive examples
    param['scale_pos_weight'] = sum_wneg/sum_wpos
    param['bst:eta'] = eta 
    param['bst:max_depth'] = depth
    param['eval_metric'] = 'auc'
    param['silent'] = 0
    param['nthread'] = number_threads
    
    # specify validations set to watch performance
    evallist  = [(dvalid,'eval'), (dtrain,'train')]

    # Train the GradientBoostingClassifier
    bst = xgb.train( param, dtrain, n_boost_iter, evallist )

    # Get the probaility output from the trained method, using the 10% for testing
    predict_train = bst.predict(dtrain)
    predict_valid = bst.predict(dvalid)

    # Select a cutoff point for assign signal and background labels
    pcut = np.percentile(predict_train,threshold)
    
    # These are the final signal and background predictions
    Yhat_train = predict_train > pcut 
    Yhat_valid = predict_valid > pcut   
    # Calc numeber of s and b TruePos and True Neg for training and validation
    s_train, b_train = higgs_lib.count_s_b(W_train,Y_train,Yhat_train)
    s_valid, b_valid = higgs_lib.count_s_b(W_valid,Y_valid,Yhat_valid)
    trial_results={}
    trial_results['loss'] = higgs_lib.inv_AMSScore(s_train,b_train)
    trial_results['valid_loss'] = higgs_lib.inv_AMSScore(s_valid,b_valid)
    trial_results['status'] = STATUS_OK
    return trial_results

# <codecell>

# Call hyperopt fmin to iterate over hyper parameters
print 'Training classifier (this may take some time!)'
hyperopt_iter = 30
trials = Trials()
best = fmin(objective, space, algo=rand.suggest, max_evals=hyperopt_iter, trials=trials)    
print best

# <codecell>

# Plot AMS train vs validataion for choices of hyper parameters
import hyper_lib
hyper_lib.plot_trials(trials)

# <codecell>

# Inspect parameter details for nth hyper parameter combination
reload(hyper_lib)
hyper_lib.which_tid(2,trials)

# <rawcell>

# Best so far:
# using log & dummies
# {'threshold': [86.0], 'depth': [3.0615393522647496], 'eta': [0.2472423296083084]}
# {'threshold': [83.0], 'depth': [6.61242116667745], 'eta': [0.23325277514869436]}
# ***
# using 
# {'threshold': [82.0], 'depth': [5.7480279001901735], 'eta': [0.17001343752893355]} used for first stack.
# {'threshold': 87.0, 'depth': 6.4814738448731894, 'eta': 0.4109641832220093}
# {'threshold': 71.0, 'depth': 11.456747161913151, 'eta': 0.4561892980077648} 100 boost, 30 rand hyperopt
# {'depth': 10.781242360313211, 'eta': 0.49086662149306215, 'pcut': 86.0}
# {'eta': 0.49086662149306215, 'pcut': 86.0} iter =500, depth = 5
# {'depth': 15.0, 'eta': 0.2773768403503458, 'pcut': 62.0}
# iter = 500
#     

# <codecell>

reload(higgs_lib)

# <codecell>

# Run xgboost with best parameters and inspect AMS
n_boost_iter = 100
threshold = 83
eta = .233
depth = 6

# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param={}
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['bst:eta'] = eta 
param['bst:max_depth'] = depth
param['eval_metric'] = 'auc'
param['silent'] = 0
param['nthread'] = number_threads

# specify validations set to watch performance
evallist  = [(dvalid,'eval'), (dtrain,'train')]

# Train
bst = xgb.train( param, dtrain, n_boost_iter, evallist )
 
# Predict 
predict_train = bst.predict(dtrain)
predict_valid = bst.predict(dvalid)

# Select a cutoff point for assign signal and background labels
pcut = np.percentile(predict_train,threshold)
 
# These are the final signal and background predictions
Yhat_train = predict_train > pcut 
Yhat_valid = predict_valid > pcut

# Calc numeber of s and b TruePos and True Neg for training and validation
s_train, b_train = higgs_lib.count_s_b(W_train,Y_train,Yhat_train)
s_valid, b_valid = higgs_lib.count_s_b(W_valid,Y_valid,Yhat_valid)
 
# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff %s' % pcut
print '   - AMS based on 90% training   sample:',higgs_lib.AMSScore(s_train,b_train)
print '   - AMS based on 10% validation sample:',higgs_lib.AMSScore(s_valid,b_valid)

# <codecell>

# Optionally write a submission csv file
# Better submissions result from applying bagging first
# Load the testing data
print 'Loading testing data'
data_test = np.loadtxt( 'test.csv', delimiter=',', skiprows=1 )
X_test = data_test[:,1:31]
I_test = list(data_test[:,0])

if USE_DUMMIES == True:
    # Make dummies
    X_test = np.hstack((X_test, higgs_lib.make_dummies(X_test)))

# Format data for xgboost
dtest = xgb.DMatrix(X_test)

# <codecell>

# Get a vector of the probability predictions which will be used for the ranking
print 'Building predictions'
Predictions_test = bst.predict(dtest)

# <codecell>

# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)
 
# Now we get the CSV data, using the probability prediction in place of the ranking
print 'Organizing the prediction results'
resultlist = []
for x in range(len(I_test)):
    resultlist.append([int(I_test[x]), Predictions_test[x], 's'*(Label_test[x]==1.0)+'b'*(Label_test[x]==0.0)])
 
# Sort the result list by the probability prediction
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[1]) 
 
# Loop over result list and replace probability prediction with integer ranking
for y in range(len(resultlist)):
    resultlist[y][1]=y+1
 
# Re-sort the result list according to the index
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[0])
 
# Write the result list data to a csv file
print 'Writing a final csv file Kaggle_higgs_prediction_output.csv'
fcsv = open('Kaggle_higgs_prediction_output.csv','w')
fcsv.write('EventId,RankOrder,Class\n')
for line in resultlist:
    theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
    fcsv.write(theline) 
fcsv.close()

# <headingcell level=2>

# Plot results for inspection

# <codecell>

plt.hist([predict_train],stacked=True, bins =30, range=[-10,8], color=[ 'Khaki'])

# <codecell>

plt.hist([Predictions_test],stacked=True, bins =30, color=[  'DarkOrange'])

