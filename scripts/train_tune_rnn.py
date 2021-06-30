#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:47:07 2021

@author: jsulloa
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, BatchNormalization, TimeDistributed
from tensorflow.keras import metrics
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#%%
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#%%
# load the dataset
synth_set = pd.read_csv('../features/resolution-med_wl-4_synth.csv')
train_set = pd.read_csv('../features/resolution-med_wl-4_train.csv')
dev_set = pd.read_csv('../features/resolution-med_wl-4_dev.csv')

train_set = synth_set.append(train_set)

X_train = train_set.loc[:,train_set.columns.str.startswith('shp')].to_numpy()
Y_train = train_set.label.astype(int).to_numpy()

X_dev = dev_set.loc[:,dev_set.columns.str.startswith('shp')].to_numpy()
Y_dev = dev_set.label.astype(int).to_numpy()

# normalize
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

# Format Dataframe as sequences
Tx = train_set.fname.value_counts()[0] # sequence length
m = len(train_set.fname.value_counts()) # number of sequences
n_feat = X_train.shape[1] # number of features
X_train = np.reshape(X_train, [m, Tx, n_feat])
Y_train = Y_train.reshape([m, Tx])

m_dev = len(dev_set.fname.value_counts()) # number of sequences
X_dev = np.reshape(X_dev, [m_dev, Tx, n_feat])
Y_dev = Y_dev.reshape([m_dev, Tx])

#%% Define the keras model
model = Sequential()
#model.add(InputLayer(input_shape=[Tx, n_feat]))
model.add(GRU(units = 128, input_shape=[Tx, n_feat], return_sequences=True))
model.add(Dropout(0.8))
model.add(BatchNormalization())
model.add(Dropout(0.83))
model.add(TimeDistributed(Dense(1, activation = "sigmoid")))
# compile the keras model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', 
              optimizer=opt, 
              metrics=[f1_m])
# fit the keras model on the dataset
hist = model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs=1000, batch_size=70, verbose=1)
model.save('../models/rnn.h5')

#%% Plot training
fig, ax = plt.subplots()
ax.plot(hist.history['f1_m'])
ax.plot(hist.history['val_f1_m'])
ax.legend(['Train', 'Val'])
ax.set_ylim([0,1])
ax.grid()
plt.show()

#%% Evaluate final model
loss, f1_score = model.evaluate(X_dev, Y_dev, verbose=0)
print('Loss and f1 score: ', loss, f1_score)

