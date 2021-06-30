#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train, tune and test statistic classifier

@author: jsulloa
"""

import pandas as pd
import matplotlib.pyplot as plt
from maad import sound, util
from sklearn import svm
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import fbeta_score, make_scorer
from classif_fcns import print_report, misclassif_idx
from sklearn.metrics import confusion_matrix, f1_score


#%% Load data
shape_train = pd.read_csv('../features/resmed_wl4/resmed_wl4_train.csv')
shape_synth = pd.read_csv('../features/resmed_wl4/resmed_wl4_synth.csv')
shape_test = pd.read_csv('../features/resmed_wl4/resmed_wl4_test.csv')
shape_dev = pd.read_csv('../features/resmed_wl4/resmed_wl4_dev.csv')

# combine real training samples with augmented samples
shape_train = shape_train.append(shape_synth)

#%% Train classifier
# assign variables
X_train = shape_train.loc[:,shape_train.columns.str.startswith('shp')]
Y_train = shape_train.label.astype(int)
shape_train.label.value_counts()

# tune classifier
clf = svm.SVC(class_weight='balanced'  )
param_grid = {'kernel': ['rbf'], 
              'gamma': uniform(0.01, 1), 
              'C': uniform(1,100)}
kfold = GroupKFold(n_splits=3)
scorer = make_scorer(fbeta_score, beta=1)
rand_search = RandomizedSearchCV(clf, param_grid, scoring=scorer, n_iter=10000,
                                 refit=True, cv=kfold, return_train_score=True, 
                                 n_jobs=-1, verbose=1)
rand_search.fit(X_train, Y_train, groups=shape_train.fname)

# print basic info
print('Best score:', rand_search.best_score_)
print('Best parameters:', rand_search.best_params_)
clf = rand_search.best_estimator_

# save model
#import joblib
#joblib.dump(clf, '../models/ALFE_resmed_wl4.joblib')

#%% Evaluate on dev dataset 
X_dev = shape_dev.loc[:,shape_dev.columns.str.startswith('shp')]
Y_dev = shape_dev.label.astype(int)
Y_pred = clf.predict(X_dev)

# Evaluate
f1_score(Y_dev, Y_pred)
confusion_matrix(Y_dev, Y_pred)
misclassified = misclassif_idx(Y_dev, Y_pred)
shape_dev.fname[misclassified['fp']].value_counts()

#%% Final evaluation on test set. 
# Note: Do not use the test set to select a model.

X_test = shape_test.loc[:,shape_test.columns.str.startswith('shp')]
Y_test = shape_test.label.astype(int)
Y_pred = clf.predict(X_test)

f1_score(Y_test, Y_pred)
confusion_matrix(Y_test, Y_pred)
misclassified = misclassif_idx(Y_test, Y_pred)
shape_test.fname[misclassified['fp']].value_counts()


#%% Visualize errors
from utils import preprocess_audio_annot
import joblib
opt_preprocess = joblib.load('../features/resmed_wl4/resmed_wl4_opt.joblib')
path_audio = '../audio_examples/'
path_annot = '../mannot/ALFE/'
fname = 'RUG12_20190819_150000.wav'
fname_audio = path_audio+fname
fname_annot = path_annot+fname[0:-3]+'txt'

output = preprocess_audio_annot(fname_audio, fname_annot, opt_preprocess)
rois, shape, Sxx, ext = output['rois'], output['shape'], output['Sxx'], output['ext']
rois_annot = output['rois_annot']

#  Set input and ground truth
Y_gt = rois.label.astype(int)
X_new = shape.loc[:,shape.columns.str.startswith('shp')]
Y_new = clf.predict(X_new)

# rois, Sxx, rois_annot, ext, Y_gt, Y_new
# Visualize, Prepare variables
t_idx = rois.min_t + (rois.max_t - rois.min_t)/2
fig, ax = plt.subplots(2,1, figsize=(10, 4))
util.plot_spectrogram(Sxx, ext, log_scale=False, ax=ax[0], colorbar=False)
util.overlay_rois(Sxx, rois_annot, extent=ext, ax=ax[0], fig=fig)
ax[1].plot(t_idx, Y_new, 'o', alpha=0.8)
ax[1].plot(t_idx, Y_gt, '*', color='red', alpha=0.5, markersize=3)
ax[1].set_ylim([-0.2,1.2])
ax[1].set_xlim([rois.min_t.iloc[0],rois.max_t.iloc[-1]])
ax[1].margins(x=0)
ax[1].grid() 
