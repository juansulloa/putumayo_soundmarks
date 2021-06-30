#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs two main steps of the workflow:
1. Split training, development and validation datasets.
2. Preprocess audio and use manual annotations to format data as structured 
tables for machine learning

@author: jsulloa@humboldt.org
"""

import pandas as pd
import joblib
from utils import listdir_pattern, batch_compute_features  # soundmark detection module

#%% Load configuration variables
opt_preprocess = {'target_fs': 24000,
                  'nperseg': 512,
                  'noverlap': 0 ,
                  'wtype': 'hann',
                  'db_range': 60,
                  'db_gain': 30,
                  # rois windowed
                  'wl': 4,
                  'step': 4,
                  'flims': (2500, 5500),
                  'tlims': (0,60),
                  # shape
                  'resolution': 'med'}
print(pd.Series(opt_preprocess))
joblib.dump(opt_preprocess, '../features/reslow_wl4_opt.joblib')

#%% Load data, and split into train, dev and test sets.
# The train set will be used to fit the model, dev to chose between models, 
# and test to assess final performance.
from sklearn.model_selection import train_test_split 
path_audio = '../audio_examples/'
flist = listdir_pattern(path_audio, '.wav')
flist_aux, flist_test = train_test_split(flist, test_size=1/3, random_state=42)
flist_train, flist_dev = train_test_split(flist_aux, test_size=1/2, random_state=42)

#%% Train: Build training dataset
# Load audio , compute features and assign labels
path_annot = '../mannot/ALFE/'
shape_train = batch_compute_features(flist_train, path_audio, path_annot, opt_preprocess)
shape_train.to_csv('../features/reslow_wl3_train.csv', index=False)

#%% Synth: Augment training samples with data augmentation
path_synth = '../data_augmentation/synth/'
flist_synth = listdir_pattern(path_synth, '.wav')
shape_synth = batch_compute_features(flist_synth, path_synth, path_synth, opt_preprocess)
shape_synth.to_csv('../features/reslow_wl3_synth.csv', index=False)

#%% Dev: Build development dataset
path_annot = '../mannot/ALFE/'
shape_dev = batch_compute_features(flist_dev, path_audio, path_annot, opt_preprocess)
shape_dev.to_csv('../features/reslow_wl3_dev.csv', index=False)

#%% Test: Build test dataset
path_annot = '../mannot/ALFE/'
shape_test = batch_compute_features(flist_test, path_audio, path_annot, opt_preprocess)
shape_test.to_csv('../features/reslow_wl3_test.csv', index=False)