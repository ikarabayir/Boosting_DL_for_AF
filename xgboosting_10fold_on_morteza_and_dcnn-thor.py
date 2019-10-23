#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
 
# Do other imports now...

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

from scipy.io import loadmat
import h5py
from sklearn.metrics import confusion_matrix
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D, Lambda
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import concatenate
import tensorflow as tf
from sklearn.model_selection import KFold
import json
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[2]:


file_name='allnewECG.csv'
data=pd.read_csv(file_name, header=None) 

eks=np.array(data)
file_name='allnewLABEL.csv'
targ=pd.read_csv(file_name, header=None) 

targ=np.array(targ)
file_name='morteza.csv'
x=pd.read_csv(file_name, header=None) 
features=np.array(x)


# In[8]:


row = features[:,:491]



num_classes=4
label2 = keras.utils.to_categorical(targ, num_classes)

x=np.array(range(8528))
np.random.seed(25)
np.random.shuffle(x)

xdata = eks[x]
fdata = row[x]
ydata = label2[x]
xdata=xdata.reshape(8528, 18286,1)


# In[9]:



kf = StratifiedKFold(n_splits=10, random_state=25)
kf.get_n_splits(xdata)


# In[10]:



fdata2=np.array(fdata)
np.sum(np.isnan(fdata2)), fdata2.shape
fdata2[fdata2 == inf] = 0
a_infs = np.where(np.isinf(fdata2))
a_infs


# In[11]:


xdata.shape, fdata2.shape, ydata.shape


# In[12]:


ydata_x=np.argmax(ydata, axis=1)

tr_x=[]
tr_f=[]
tr_y=[]

te_x=[]
te_f=[]
te_y=[]

for train_index, test_index in kf.split(xdata,ydata_x):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xdata[train_index], xdata[test_index]
    f_train, f_test = fdata2[train_index], fdata2[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]
    
    tr_x.append(X_train)
    te_x.append(X_test)
    tr_f.append(f_train)
    te_f.append(f_test)
    tr_y.append(y_train)
    te_y.append(y_test)


# In[14]:


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


# In[52]:


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.001
    if epoch > 75:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 25:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# In[58]:


from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten, SeparableConv1D, Reshape

def build_model():

    input_img = Input(shape=(18286,1), name='ImageInput')
    x = Conv1D(320, 3, activation='relu', padding='same', strides=1, name='Conv1_1')(input_img)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling1D(2, name='pool1')(x)
    x = Dropout(0.3, name='dropout1')(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)


    x = Flatten(name='flatten')(x)

    x = Dense(4, activation='softmax', name='fc3')(x)
    model = Model(inputs=input_img, outputs=x)
    adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
   
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# In[ ]:


#############thor######################
layer_name = 'flatten'

predictions_train=[]
predictions_test=[]

predictions_test_xg=[]

true_class=[]


cf_all=[]
acc_all=[]
f1_all=[]
acc=[]

ep=30
est=1500

for i in range(10):
    print ('Model'+' '+str(i)+' '+'starts...')
    model = build_model()
    for epoch in range(ep):
        print ('Model'+' '+str(i)+' '+'epoch...' +str(epoch))
        K.set_value(model.optimizer.lr, lr_schedule(epoch))
        history = model.fit(tr_x[i], tr_y[i], epochs=1, batch_size=16, validation_data=(te_x[i], te_y[i]))    
    intermediate_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer(layer_name).output)
    
    print ('creating features')
    intermediate_output_train = intermediate_layer_model.predict(tr_x[i])
    intermediate_output_test = intermediate_layer_model.predict(te_x[i])
    
    a=np.concatenate((intermediate_output_train, tr_f[i] ), axis=1)
    b=np.concatenate((intermediate_output_test, te_f[i] ), axis=1)
    
    print (a.shape)
    print (b.shape)
    predictions_train.append(a)
    predictions_test.append(b)
    
    print ('Model XGBClassifier'+' '+str(i)+' '+'starts...')
    modelxgb = XGBClassifier(n_estimators=est, seed=25)
    modelxgb.fit(a, np.argmax(tr_y[i],axis=1))

    test_predict = modelxgb.predict(b)
    predictions_test_xg.append(test_predict)
    true_class.append(te_y[i])
    print (confusion_matrix(np.argmax(te_y[i],axis=1), test_predict))
    print (f1_custom(np.argmax(te_y[i],axis=1), test_predict))
    print (accuracy_cus(np.argmax(te_y[i],axis=1), test_predict))
    cf_all.append(confusion_matrix(np.argmax(te_y[i],axis=1), test_predict))
    f1_all.append(f1_custom(np.argmax(te_y[i],axis=1), test_predict)[4])
    acc_all.append(accuracy_cus(np.argmax(te_y[i],axis=1), test_predict))
    accuracy = accuracy_score(np.argmax(te_y[i],axis=1), test_predict)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    acc.append(accuracy)
    K.clear_session()
    
#############thor######################


# In[ ]:




