# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Averaging stacker

# <codecell>

import platform

if platform.system() == 'Darwin':
    xgboost_path = '/Users/andyh_mac/xgboost/xgboost-master/python'
    number_threads = 8
    data_dir = '/Users/andyh_mac/Desktop/Analytics/Higgs/data_std_2split/'
elif platform.system() == 'Linux':
    xgboost_path = '/home/ubuntu/xgboost-master/python'
    number_threads = 32
    data_dir = '/mnt2/Higgs/data_std_2split/'
else:
    print "Don't know parameters for this system: %s" % platform.system()

# <codecell>

from hyperopt import fmin, rand, tpe, space_eval, hp, Trials
from hyperopt import STATUS_OK, STATUS_FAIL
from IPython.parallel import Client
from pymongo import MongoClient
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.preprocessing import normalize
import higgs_lib
import hyper_lib
import numpy as np
import math
import pandas as pd
import pickle
 
# Load training data from Mongo
print 'Loading training data from mongo higgs.data'
client = MongoClient()
db = client.higgs

# Load X train,valid from Mongo
def get_db(type):
    # Find right record type
    docs = db.data.find( { '$and' : [{'type': 'train'}, {'id': {'$regex': '01|02|03|21|23'}}] })   #use first 3 datasets
    # Retreive the data set
    train_data = []
    valid_data = []
    for d in docs:
        train_data.append(d['stack_train'])
        valid_data.append(d['stack_valid'])
    # Convert train data to array
    dim0= len(train_data[0])
    dim1 = len(train_data)
    X_train = np.empty((dim0,dim1))
    for j,d in enumerate(train_data):
        c = np.array(d)
        X_train[:,j]=c
    # Convert validation data to array
    dim0= len(valid_data[0])
    dim1 = len(valid_data)
    X_valid = np.empty((dim0,dim1))
    for j,d in enumerate(valid_data):
        c = np.array(d)
        X_valid[:,j]=c
    return (X_train, X_valid)
X_train, X_valid = get_db('train')

# <codecell>

# Load Higgs csv files to get Y, W
print 'Assigning data to numpy arrays.'
Y_train = np.loadtxt( data_dir + 'Y_train_2.csv', delimiter=',', skiprows=1 )
W_train = np.loadtxt( data_dir + 'W_train_2.csv', delimiter=',', skiprows=1 )
Y_valid = np.loadtxt( data_dir + 'Y_valid_2.csv', delimiter=',', skiprows=1 )
W_valid = np.loadtxt( data_dir + 'W_valid_2.csv', delimiter=',', skiprows=1 )

# <codecell>

# Define objective (loss) function
space = [ hp.quniform('threshold',60,90,1)]
METHOD = 'mean'
def my_obj(args):
    threshold = args
    threshold = int(threshold[0])

    # Apply a stacking method
    if METHOD == 'mean':
        prob_predict_train = np.apply_along_axis(np.mean,1,X_train)
        prob_predict_valid = np.apply_along_axis(np.mean,1,X_valid) 
    elif METHOD == 'geomean':
        def p(x):
            z=np.apply_along_axis(np.cumproduct,1,x)[:,-1]
            return np.power(z, 1.0/x.shape[1])
        prob_predict_train = p(X_train)
        prob_predict_valid = p(X_valid)
    else:
        print 'METHOD must be mean or prob'
        stop()
        
    # Choose cut point
    pcut = np.percentile(prob_predict_train,threshold)
     
    # These are the final signal and background predictions
    Yhat_train = prob_predict_train > pcut 
    Yhat_valid = prob_predict_valid > pcut

    # Calc numeber of s and b TruePos and True Neg for training and validation
    s_train, b_train = higgs_lib.count_s_b(W_train,Y_train,Yhat_train)
    s_valid, b_valid = higgs_lib.count_s_b(W_valid,Y_valid,Yhat_valid)

   
    # Now calculate the invers AMS scores
    def inv_AMSScore(s,b):
        try:
            inv_ams = 1/math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
        except:
            inv_ams= 1
            pass
        return inv_ams
    trial_results={}
    trial_results['loss'] = inv_AMSScore(s_train,b_train)
    trial_results['valid_loss'] = inv_AMSScore(s_valid,b_valid)
    trial_results['status'] = STATUS_OK
    return trial_results
    

# <codecell>

# Set up hyperopt
trials= Trials()
hyperopt_iter =30
print 'Training classifier (this may take some time!)'
best = fmin(fn=my_obj, space=space, algo=rand.suggest, max_evals=hyperopt_iter, trials=trials)    
print best

# <codecell>

# Plot trial results
hyper_lib.plot_trials(trials)

# <codecell>

# Inspect trial details
hyper_lib.which_tid(25,trials)

# <codecell>

# Predict train and valid classes
# Select threshold found in hyperopt
best = {'threshold': 84.0}
# Stack the inputs
print 'Training classifier'
threshold = int(best['threshold'])
if METHOD == 'mean':
    prob_predict_train = np.apply_along_axis(np.mean,1,X_train)
    prob_predict_valid = np.apply_along_axis(np.mean,1,X_valid) 
elif METHOD == 'geomean':
    def p(x):
        z=np.apply_along_axis(np.cumproduct,1,x)[:,-1]
        return np.power(z, 1.0/x.shape[1])
    prob_predict_train = p(X_train)
    prob_predict_valid = p(X_valid)
else:
    print 'METHOD must be mean or prob'
    stop()
        
pcut = np.percentile(prob_predict_train,threshold)
 
# This are the final signal and background predictions
Yhat_train = prob_predict_train > pcut 
Yhat_valid = prob_predict_valid > pcut
 
# Calc numeber of s and b TruePos and True Neg for training and validation
s_train, b_train = higgs_lib.count_s_b(W_train,Y_train,Yhat_train)
s_valid, b_valid = higgs_lib.count_s_b(W_valid,Y_valid,Yhat_valid)
 
# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=',pcut
print '   - AMS based on 90% training   sample:',higgs_lib.AMSScore(s_train,b_train)
print '   - AMS based on 10% validation sample:',higgs_lib.AMSScore(s_valid,b_valid)

# <codecell>

# Plot histogram of class probabilities for inspection
plt.hist([prob_predict_valid[Y_valid==0.0],prob_predict_valid[Y_valid==1.0]], bins = 30, stacked=True,  color=[ 'Khaki', 'DarkOrange'])
         

# <codecell>

# Now load the testing data, storing the data (X) and index (I)
def get_db_test():
    # Find right record type
    docs = db.data.find( { '$and' : [{'type': 'test'}, {'id': {'$regex': '01|02|03|21|23|11|14'}}] })  #use first 3 datasets
    # Retreive the data set
    test_data = []
    for d in docs:
        test_data.append(d['stack_test'])
    # Convert test data to array
    dim0= len(test_data[0])
    dim1 = len(test_data)
    X_test = np.empty((dim0,dim1))
    for j,d in enumerate(test_data):
        c = np.array(d)
        X_test[:,j]=c
    return (X_test)

print 'Loading testing data from mongo'
data_test = np.loadtxt( 'test.csv', delimiter=',', skiprows=1 )
X_raw = get_db_test()
X_test = np.zeros(X_raw.shape)
def proba(x): return 1.0/(1 + exp(-x))
for i in range(X_raw.shape[1]):
    if np.min(X_raw[:,i]) < 0 or np.max(X_raw[:,i]) > 1:
        X_test[:,i] = apply_along_axis(proba,0,X_raw[:,i])
    else:
        X_test[:,i] = X_raw[:,i]
I_test = list(data_test[:,0])
 
# Get a vector of the probability predictions which will be used for the ranking
print 'Building predictions'
if METHOD == 'mean':
    Predictions_test = np.apply_along_axis(np.mean,1,X_test)
elif METHOD == 'geomean':
    def p(x):
        z=np.apply_along_axis(np.cumproduct,1,x)[:,-1]
        return np.power(z, 1.0/x.shape[1])
    Predictions_test = p(X_test)
else:
    print 'METHOD must be mean or prob'
    stop()

# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)

# <codecell>

# Now get the CSV data, using the probability prediction in place of the ranking
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

# <codecell>

# Plot histogram of Predictions for inspection
plt.figure(2)
sel1 = np.array(Label_test) == 1.0
sel0 = np.logical_not(sel1)
plt.hist( [np.array(Predictions_test)[sel0 ], np.array(Predictions_test)[sel1]], stacked=True, bins=30,color=['Khaki','DarkOrange'])

# <codecell>

# Compare number of predicted positives for test with number of postives in train
print 'number of postives for test: %s' % sum(Label_test)
print 'number of postives for train(normalized to test): %s' % str(sum(Yhat_train)*550000/(X_train.shape[0]))

