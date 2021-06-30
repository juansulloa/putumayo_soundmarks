#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict new instances with a trained model.

@author: jsulloa
"""
import os
import numpy as np
import pandas as pd
import joblib
from utils import (listdir_pattern, preprocess_audio_annot, batch_compute_features)

#%% Load preprocessing parameters and model
opt_preprocess = joblib.load('../features/resmed_wl3/resmed_wl3_opt.joblib')  # preprocessing options
model = joblib.load('../models/ALFE_resmed_wl3_F1_0.873.joblib')  # trained model
path_audio = '../audio_examples/'  # path to directory with audio samples

#%% Batch predict multiple instances
flist = listdir_pattern(path_audio, '.wav')
# flist = flist[0:10]  # for testing purposes
shape = batch_compute_features(flist, path_audio, path_annot=None, opt=opt_preprocess)

# save features
shape.to_csv('../features/new_data/shape_test.csv')

# svm
X_new = shape.loc[:,shape.columns.str.startswith('shp')]
Y_new = model.predict(X_new)
pred_rois = pd.DataFrame({'fname': shape['fname'], 'score': Y_new})
pred_file = pred_rois.groupby('fname').sum()

# print results and save data
print(pred_file.sort_values('score').tail(50))
pred_rois.to_csv('../inference_new_data/predictions_rois.csv')
pred_file.to_csv('../inference_new_data/predictions_file.csv')

