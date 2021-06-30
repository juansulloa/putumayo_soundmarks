#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:57:38 2021

@author: jsulloa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras import models
from utils import (listdir_pattern, 
                                preprocess_audio_annot, 
                                batch_compute_features)
import config_vars

#%% Load preprocessing parameters and model
opt_preprocess = config_vars.opt_preprocess
model = joblib.load('../models/ALFE_shape_med.joblib')
model = models.load_model('../models/rnn.h5', custom_objects={'f1_m': f1_m, 'K': K})

#%% Load audio, preprocess and predict a single instance
fname_audio = '../audio_examples/RUG03_20190810_230000.wav'
output = preprocess_audio_annot(fname_audio, fname_annot=None, opt=opt_preprocess)
shape = output['shape']

# svm
X_new = shape.loc[:,shape.columns.str.startswith('shp')]
Y_new = model.predict(X_new)

# rnn
X_new = shape.loc[:,shape.columns.str.startswith('shp')].to_numpy()
Tx = 15 # sequence length
n_feat = X_new.shape[1] # number of features
X_new = np.reshape(X_new, [1, Tx, n_feat])
Y_new = model.predict(X_new)[0]

# Visualize
from maad import util
Sxx = output['Sxx']
ext = output['ext']
rois = output['rois']
# Visualize, Prepare variables
t_idx = rois.min_t + (rois.max_t - rois.min_t)/2
fig, ax = plt.subplots(2,1, figsize=(10, 4))
util.plot_spectrogram(Sxx, ext, log_scale=False, ax=ax[0], colorbar=False)
#util.overlay_rois(Sxx, rois_annot, extent=ext, ax=ax[0], fig=fig)
ax[1].plot(t_idx, Y_new, 'o', alpha=0.8)
#ax[1].plot(t_idx, Y_gt, '*', color='red', alpha=0.5, markersize=3)
ax[1].set_ylim([-0.2,1.2])
ax[1].set_xlim([rois.min_t.iloc[0],rois.max_t.iloc[-1]])
ax[1].margins(x=0)
ax[1].grid() 

#%% Batch predict multiple instances
path_audio = '../audio_examples/'
flist = listdir_pattern(path_audio, '.wav')
flist = flist[0:10]
shape = batch_compute_features(flist, path_audio, path_annot=None, opt=opt_preprocess)

# svm
X_new = shape.loc[:,shape.columns.str.startswith('shp')]
Y_new = model.predict(X_new)

# rnn
X_new = shape.loc[:,shape.columns.str.startswith('shp')].to_numpy()
m = len(flist)
Tx = 15 # sequence length
n_feat = X_new.shape[1] # number of features
X_new = np.reshape(X_new, [m, Tx, n_feat])
Y_new = model.predict(X_new)
