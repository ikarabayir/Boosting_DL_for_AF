#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division

from keras import backend as K
import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from keras.layers import concatenate
import pickle
import pandas as pd
import glob
import numpy as np
from numpy import inf
import scipy.stats
from scipy.io import loadmat
import h5py
from sklearn.metrics import confusion_matrix
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D, Lambda
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import concatenate
import tensorflow as tf
import json
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import xgboost as xgb

def f1_custom(y_true, y_pred):
    """f1 metric.
    Computes the f1 over the whole batch using threshold_value.
    """

    cf=confusion_matrix(y_true, y_pred)
    f1_0=2*cf[0][0]/(np.sum(cf[0])+np.sum(cf[:,0]))
    f1_1=2*cf[1][1]/(np.sum(cf[1])+np.sum(cf[:,1]))
    f1_2=2*cf[2][2]/(np.sum(cf[2])+np.sum(cf[:,2]))
    f1_3=2*cf[3][3]/(np.sum(cf[3])+np.sum(cf[:,3]))
    fmean=np.mean([f1_0, f1_1, f1_2])
    return ([f1_0, f1_1, f1_2, f1_3, fmean])

def f1_custom2(y_true, y_pred):
    """f1 metric.
    Computes the f1 over the whole batch using threshold_value.
    """

    cf=tf.confusion_matrix(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1))
    f1_0=2*cf[0][0]/(K.sum(cf[0])+K.sum(cf[:,0]))+ K.epsilon()
    f1_1=2*cf[1][1]/(K.sum(cf[1])+K.sum(cf[:,1]))+ K.epsilon()
    f1_2=2*cf[2][2]/(K.sum(cf[2])+K.sum(cf[:,2]))+ K.epsilon()
    f1_3=2*cf[3][3]/(K.sum(cf[3])+K.sum(cf[:,3]))+ K.epsilon()
    fmean=K.mean(tf.convert_to_tensor([f1_0+ K.epsilon(), f1_1+ K.epsilon(), f1_2+ K.epsilon()]))
    return fmean

def accuracy_cus(y_true, y_pred):

    cf=confusion_matrix(y_true, y_pred)
    acc_0=cf[0][0]/np.sum(cf[0])
    acc_1=cf[1][1]/np.sum(cf[1])
    acc_2=cf[2][2]/np.sum(cf[2])
    acc_3=cf[3][3]/np.sum(cf[3])
    return ([acc_0, acc_1, acc_2, acc_3])

def feature_model4(first, X,Y,D, X2, Y2, D2, est_feat):
    modelxgb = XGBClassifier(est=est_feat, seed=25 )
    ilk=np.sum(np.array(D[:first]).flatten())
    xgb = modelxgb.fit(np.concatenate(X, axis=0)[:ilk], np.concatenate(Y, axis=0)[:ilk])
    
    test_predict = modelxgb.predict_proba(np.concatenate(X, axis=0))
    
    mean_tr=[]
    std_tr=[]
    std_mean_tr=[]
    
    mean_tr2=[]
    std_tr2=[]
    std_mean_tr2=[]
    for i in range(len(D)):
        m = np.mean(test_predict[np.sum(np.array(D).flatten()[:i]):np.sum(np.array(D).flatten()[:i+1])], axis=0)
        s = np.std(test_predict[np.sum(np.array(D).flatten()[:i]):np.sum(np.array(D).flatten()[:i+1])], axis=0)
        s_m = s/m
        mean_tr.append(m)
        std_tr.append(s)
        std_mean_tr.append(s_m)
        
        m2 = np.mean(X[i], axis=0)
        s2 = np.std(X[i], axis=0)
        s_m2 = s2/m2
        mean_tr2.append(m2)
        std_tr2.append(s2)
        std_mean_tr2.append(s_m2)        
        
        
        
    test_predict = modelxgb.predict_proba(np.concatenate(X2, axis=0))
    
    mean_te=[]
    std_te=[]
    std_mean_te=[]
    
    mean_te2=[]
    std_te2=[]
    std_mean_te2=[]
    
    for i in range(len(D2)):
        m = np.mean(test_predict[np.sum(np.array(D2).flatten()[:i]):np.sum(np.array(D2).flatten()[:i+1])], axis=0)
        s = np.std(test_predict[np.sum(np.array(D2).flatten()[:i]):np.sum(np.array(D2).flatten()[:i+1])], axis=0)
        s_m = s/m
        mean_te.append(m)
        std_te.append(s)
        std_mean_te.append(s_m)
        
        
        m2 = np.mean(X2[i], axis=0)
        s2 = np.std(X2[i], axis=0)
        s_m2 = s2/m2
        mean_te2.append(m2)
        std_te2.append(s2)
        std_mean_te2.append(s_m2)       
        
    v= [mean_tr, std_tr, std_mean_tr, mean_te, std_te, std_mean_te, 
        mean_tr2, std_tr2, std_mean_tr2, mean_te2, std_te2, std_mean_te2]
        
    f1 = np.concatenate([v[0], v[1], v[2]],axis=1)
    f2 = np.concatenate([v[3], v[4], v[5]],axis=1)
    
    f3 = np.concatenate([v[6], v[7], v[8]],axis=1)
    f4 = np.concatenate([v[9], v[10], v[11]],axis=1)
    
    feat1=np.concatenate([f1,f3],axis=1)
    feat2=np.concatenate([f2,f4],axis=1)
    
    return (feat1, feat2, xgb)




file_name='mydataREP.csv'
dataREP=pd.read_csv(file_name, header=None) 
dataREP=np.array(dataREP)

file_name='mydataprepy.csv'
dataREPy=pd.read_csv(file_name, header=None) 
dataREPy=np.array(dataREPy)

file_name='mydataprepx.csv'
dataREPx=pd.read_csv(file_name, header=None) 
dataREPx=np.array(dataREPx)


file_name='allnewLABEL.csv'
targ=pd.read_csv(file_name, header=None) 

targ=np.array(targ)
file_name='morteza.csv'
x=pd.read_csv(file_name, header=None) 
features=np.array(x)



seg_data_x=[]
seg_data_y=[]
for i in range(8528):
    seg_data_x.append(dataREPx[np.sum(dataREP[:i]):np.sum(dataREP[:i+1])])
    seg_data_y.append(dataREPy[np.sum(dataREP[:i]):np.sum(dataREP[:i+1])])
    
    
row = np.concatenate([features[:,:199], features[:,241:]], axis=1)

num_classes=4
label2 = keras.utils.to_categorical(targ, num_classes)

x=np.array(range(8528))
np.random.seed(25)
np.random.shuffle(x)


seg_data_x2=[]
seg_data_y2=[]
dataREP2=[]

for i in range(8528):
    seg_data_x2.append(seg_data_x[x[i]])
    seg_data_y2.append(seg_data_y[x[i]])
    dataREP2.append(dataREP[x[i]])
    

fdata = row[x]
ydata = label2[x]


kf = StratifiedKFold(n_splits=10, random_state=25)
kf.get_n_splits(fdata)


fdata2=np.array(fdata)
np.sum(np.isnan(fdata2)), fdata2.shape
fdata2[fdata2 == inf] = 0
a_infs = np.where(np.isinf(fdata2))


ydata_x=np.argmax(ydata, axis=1)
##########################
tr_x_seg=[]
tr_y_seg=[]
tr_dim_seg=[]


tr_f=[]
tr_y=[]
##########################
te_x_seg=[]
te_y_seg=[]
te_dim_seg=[]


te_f=[]
te_y=[]



for train_index, test_index in kf.split(fdata,ydata_x):
    sX_train=[]
    sY_train=[]
    sD_train=[]
    
    for i in range(train_index.shape[0]):
        sX_train.append(seg_data_x2[train_index[i]])
        sY_train.append(seg_data_y2[train_index[i]])
        sD_train.append(dataREP2[train_index[i]])
    sX_test=[]
    sY_test=[]
    sD_test=[]
    
    for i in range(test_index.shape[0]):
        sX_test.append(seg_data_x2[test_index[i]])
        sY_test.append(seg_data_y2[test_index[i]])
        sD_test.append(dataREP2[test_index[i]])
   
    f_train, f_test = fdata2[train_index], fdata2[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]
    
    tr_x_seg.append(sX_train)
    te_x_seg.append(sX_test)
    tr_y_seg.append(sY_train)
    te_y_seg.append(sY_test)
    tr_dim_seg.append(sD_train)
    te_dim_seg.append(sD_test)
    
    tr_f.append(f_train)
    te_f.append(f_test)
    tr_y.append(y_train)
    te_y.append(y_test)   
    
    
predictions_train=[]
predictions_test=[]

for i in range(10):
    file_test='test_fold_'+str(i)+'.csv'
    file_train='train_fold_'+str(i)+'.csv'
    predictions_test.append(np.loadtxt(file_test, delimiter=","))
    predictions_train.append(np.loadtxt(file_train, delimiter=","))
    
    


#first 128 comes from CNN, then after that 198 based on base level and last 251 also based on base level
# or exclude between 327 and 367

i=2

ilk=1535
est_feat=500

egi=np.concatenate([predictions_train[i][:,:198+128], predictions_train[i][:,240+128:]], axis=1)
tnew=np.concatenate([predictions_test[i][:,:198+128], predictions_test[i][:,240+128:]], axis=1)

trfeat, tefeat, model_feat = feature_model4(ilk, tr_x_seg[i], tr_y_seg[i], tr_dim_seg[i],
                                           te_x_seg[i], te_y_seg[i], te_dim_seg[i], est_feat )


egi2= np.concatenate([egi,trfeat], axis=1)
tnew2= np.concatenate([tnew,tefeat], axis=1)

def xgb_evaluate(max_depth, gamma, colsample_bytree, n_estimators, min_child_weight, learning_rate):
    params = {'max_depth': int(max_depth),
              'subsample': 0.8,
              'seed': 25,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree,
              'n_estimators' :int(n_estimators),
              'min_child_weight' : min_child_weight,
              'learning_rate' :learning_rate 
             }
    i=2
    # Used around 1000 boosting rounds in the full model
    modelxgb = XGBClassifier()
    modelxgb.set_params(**params)
    xgb = modelxgb.fit(egi2, np.argmax(tr_y[i],axis=1))
    test_predict = modelxgb.predict(tnew2)
    
    print (f1_custom(np.argmax(te_y[i],axis=1), test_predict)[-1])
    print (modelxgb.get_params())
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return f1_custom(np.argmax(te_y[i],axis=1), test_predict)[-1]


xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9),
                                             'n_estimators': (300,500),
                                             'min_child_weight':(0, 20),
                                             'learning_rate':(0.05, 0.1),
                                            })
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=3, n_iter=40, acq='ei')


xgb_bo.max

#founded::::::::::
best_params={'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.6452037332243158, 'gamma': 0.07867994069460738, 'learning_rate': 0.05972291926962713, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 0.8277318755737273, 'missing': None, 'n_estimators': 301, 'n_jobs': 1, 'nthread': None, 'objective': 'multi:softprob', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': 25, 'silent': None, 'subsample': 0.8, 'verbosity': 1}



n_estimator=[0]

learn=[0]

ilk=1535
est_feat=500

importance=[]
gs_predictions_test_xg=[]
gs_cf_all=[]
gs_f1_all=[]
gs_acc_all=[]
gs_acc=[]


for j in range(1):
    for k in range(1):
        for i in range(10):
            print ('FeatureModel XGBClassifier __for estimator='+' '+'__for lr=' + '__fold number='+ str(i) + '__starts...')
            
            egi=np.concatenate([predictions_train[i][:,:198+128], predictions_train[i][:,240+128:]], axis=1)
            tnew=np.concatenate([predictions_test[i][:,:198+128], predictions_test[i][:,240+128:]], axis=1)
            
            trfeat, tefeat, model_feat = feature_model4(ilk, tr_x_seg[i], tr_y_seg[i], tr_dim_seg[i],
                                                       te_x_seg[i], te_y_seg[i], te_dim_seg[i], est_feat )
            
            
            print ('Model XGBClassifier __for estimator='+' '+'__for lr=' + '__fold number='+ str(i) + '__starts...')

            egi2= np.concatenate([egi,trfeat], axis=1)
            tnew2= np.concatenate([tnew,tefeat], axis=1)
            
            modelxgb = XGBClassifier()
            modelxgb.set_params(**best_params)
            
            xgb = modelxgb.fit(egi2, np.argmax(tr_y[i],axis=1))
            test_predict = modelxgb.predict(tnew2)
            
            gs_predictions_test_xg.append(test_predict)
            print (confusion_matrix(np.argmax(te_y[i],axis=1), test_predict))
            print (f1_custom(np.argmax(te_y[i],axis=1), test_predict))
            print (accuracy_cus(np.argmax(te_y[i],axis=1), test_predict))
            gs_cf_all.append(confusion_matrix(np.argmax(te_y[i],axis=1), test_predict))
            gs_f1_all.append(f1_custom(np.argmax(te_y[i],axis=1), test_predict)[4])
            gs_acc_all.append(accuracy_cus(np.argmax(te_y[i],axis=1), test_predict))
            accuracy = accuracy_score(np.argmax(te_y[i],axis=1), test_predict)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            gs_acc.append(accuracy)
            importance.append(xgb.feature_importances_)

            


# In[ ]:




