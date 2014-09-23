# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Bagging for xgboost

# <markdowncell>

# test data format:

# <rawcell>

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

# Bagging vars
bag_ratio =.8
bag_iter = 100
models=[]

# Set xgboost params
best = {'threshold': 86, 'depth': 7, 'eta': 0.2}
boost_iter = 100
param = {}
param['objective'] = 'binary:logitraw'  #raw coef not prob. why?
param['bst:eta'] = best['eta'] 
param['bst:max_depth'] = int(best['depth'])
param['eval_metric'] = 'auc'
param['silent'] = 0
param['nthread'] = number_threads

# Pick random seed.
np.random.seed(42)

# Start bagging
for i in range(bag_iter):
    # Select bag sampl

    # Random number for training/validation splitting
    r =np.random.rand(X_train.shape[0])
    
    # Put Y(truth), X(data), W(weight), and I(index) into their own arrays
    print 'bag iteration: %s' % i
    # First 90% are training
    Y_train_bag = Y_train[r<bag_ratio]
    X_train_bag = X_train[r<bag_ratio]
    W_train_bag = W_train[r<bag_ratio]

    # Set xgboost data
    weight = W_train_bag * (float(higgs_lib.TRAINING_LENGTH) / len(Y_train_bag))
    dtrain = xgb.DMatrix(X_train_bag,label=Y_train_bag, weight = weight)
    sum_wpos = sum( weight[i] for i in range(len(Y_train_bag)) if Y_train_bag[i] == 1.0  )
    sum_wneg = sum( weight[i] for i in range(len(Y_train_bag)) if Y_train_bag[i] == 0.0  )
    
    # specify iteration dependent params
    evallist  = [(dvalid,'eval'), (dtrain,'train')]
    param['scale_pos_weight'] = sum_wneg/sum_wpos # scale weight of positive examples
    
    # Make model for this data sample
    mdl = xgb.train( param, dtrain, boost_iter, evallist )
    models.append(mdl)


# <codecell>

# Define bagging prediction fn
def bag_predict(models, data):
    predicted = np.empty((data.num_row(),len(models)))
    predicted[:] = np.NAN
    for i,m in enumerate(models):
        predicted[:,i] = m.predict(data)
    return np.apply_along_axis(np.mean,1,predicted)

# <codecell>

# Make predictions with bagging models
# Use full training set, not bag ratioed
dtrain = xgb.DMatrix(X_train,label=Y_train, weight = weight)
predict_train = bag_predict(models,dtrain)
predict_valid = bag_predict(models,dvalid)

# Select a cutoff point for assign signal and background labels
pcut = np.percentile(predict_train,best['threshold'])

# This are the final signal and background predictions
Yhat_train = predict_train > pcut 
Yhat_valid = predict_valid > pcut
 
# Calc numeber of s and b TruePos and True Neg for training and validation
s_train, b_train = higgs_lib.count_s_b(W_train,Y_train,Yhat_train)
s_valid, b_valid = higgs_lib.count_s_b(W_valid,Y_valid,Yhat_valid)
 
# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=',pcut
print '   - AMS based on 90% training   sample:',higgs_lib.AMSScore(s_train,b_train)
print '   - AMS based on 10% validation sample:',higgs_lib.AMSScore(s_valid,b_valid)
 

# <codecell>

# Now we load the testing data, storing the data (X) and index (I)
print 'Loading testing data'
data_test = np.loadtxt( 'test_log.csv', delimiter=',', skiprows=1 )
X_test = data_test[:,1:]
I_test = list(data_test[:,0])

if USE_DUMMIES == True:
    # Make dummies
    X_test = np.hstack((X_test, make_dummies(X_test)))
    

# <codecell>

dtest = xgb.DMatrix(X_test)
 
# Get a vector of the probability predictions which will be used for the ranking
print 'Building predictions'
Predictions_test = bag_predict(models,dtest)
# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)
 

# <codecell>

# Optional
# Write the result list data to a submission csv file
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
# Write the file
print 'Writing a final csv file Kaggle_higgs_prediction_output.csv'
fcsv = open('Kaggle_higgs_prediction_output.csv','w')
fcsv.write('EventId,RankOrder,Class\n')
for line in resultlist:
    theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
    fcsv.write(theline) 
fcsv.close()

# <codecell>

# Convert from logit raw format to probabilities
print 'Building train test predictions for stacking'
def proba(x): return 1.0/(1 + exp(-x))
train_stack_prob = apply_along_axis(proba,0,predict_train)  # convert to probabillities to mix with other model outputs
valid_stack_prob = apply_along_axis(proba,0,predict_valid)  # convert to probabillities to mix with other model outputs
test_stack_prob = apply_along_axis(proba,0,Predictions_test)  # convert to probabillities to mix with other model outputs

# <codecell>

plt.hist([test_stack_prob],stacked=True, bins =30, color=[ 'Khaki'])

# <codecell>

# Save X predictions. Validation predictions and Test predictions along with param details to MongoDB
# Stacking algo will pull data from MongoDB
# Saving as two documents to avoid 15MB Mongo limit
client = MongoClient()
db = client.higgs
# save predictions for stacking
run_id = '23'
comment ="run 3 of log,dummy bagged xgb"
doc = {}
doc['id']= run_id
doc['type']= 'train'
doc['comment']=comment
doc['algo']='xgb'
doc['boost_iter']=boost_iter
doc['depth']= best['depth']
doc['eta'] = best['eta']
doc['threshold'] = best['threshold']
doc['bag_iter'] = bag_iter
doc['bag_ratio']= bag_ratio
doc['stack_train']= train_stack_prob.tolist()
doc['stack_valid']= valid_stack_prob.tolist()
print 'Saving train data set to mongo db in higgs.data'
db.data.insert(doc)
doc = {}
doc['id']= run_id
doc['type']= 'test'
doc['comment']=comment
doc['algo']='xgb'
doc['boost_iter']=boost_iter
doc['depth']= best['depth']
doc['eta'] = best['eta']
doc['threshold'] = best['threshold']
doc['bag_iter'] = bag_iter
doc['bag_ratio']= bag_ratio
doc['stack_test']= test_stack_prob.tolist()
print 'Saving test data set to mongo db in higgs.data'
db.data.insert(doc)

# <headingcell level=2>

# Compare train and test coefficients

# <codecell>

print 'number of postives for test: %s' % sum(Label_test)
print 'number of postives for train(normalized to test): %s' % str(sum(Yhat_train)*550000/(X_train.shape[0]))

# <codecell>

plt.hist([predict_valid],stacked=True, bins =30, color=[ 'Khaki'], range=[-10,8])

# <codecell>

plt.hist(Predictions_test, bins = 30, range=[-10,8], color=[ 'DarkOrange'])

