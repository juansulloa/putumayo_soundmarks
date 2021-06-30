#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble of functions to facilitate the preprocessing of audio and annotations

@author: jsulloa
"""

import numpy as np
import pandas as pd
from maad import util, sound, features
from os import listdir

def listdir_pattern(path_dir, ends_with=None):
    """
    Wraper function from os.listdir to include a filter to search for patterns
    
    Parameters
    ----------
        path_dir: str
            path to directory
        ends_with: str
            pattern to search for at the end of the filename
    Returns
    -------
    """
    flist = listdir(path_dir)

    new_list = []
    for names in flist:
        if names.endswith(ends_with):
            new_list.append(names)
    return new_list


def rois_windowed(wl, step, tlims=(0, 60), flims=(0, 22050), rois_annot=None, tn=None, fn=None):
    """
    Discretize audio signal into multiple segments and use manual annotations to label 
    rois.
    
    Parameters
    ----------
    wl : float
        Window length in seconds.
    step : float
        Step for winowed roi in seconds
    tlims : tuple, optional
        Temporal limits to create rois in seconds. The default is (0, 60).
    flims : tuple, optional
        Frequency limits to create rois in Hertz. The default is (0, 22050).
    rois_annot : pandas DataFrame
        regions of interest with annotations.

    Returns
    -------
    rois : pandas Dataframe

    """
    # init rois windowed
    rois = pd.DataFrame({'min_t': np.arange(tlims[0], tlims[1]-wl+1, step),
                         'max_t': np.arange(tlims[0]+wl, tlims[1]+1, step),
                         'min_f': flims[0],
                         'max_f': flims[1],
                         'label': '0'})
    
    rois = util.format_features(rois, tn, fn)
    
    # if no annotations provided, return windowed rois
    if rois_annot is None:
        return rois
    
    # if provided, add rois labels
    for idx, row in rois_annot.iterrows():
        if wl==step:
            idx_min_t = (rois.min_t - row.min_t).abs().idxmin()
            idx_max_t = (rois.max_t - row.max_t).abs().idxmin()
            rois.loc[idx_min_t:idx_max_t, 'label'] = row.label
        else:
            print('TODO: correct assignment of labels when wl < step')
            #idx_min_t = ((row.min_t - wl*0.5) - rois.min_t)<=0
            #idx_max_t = ((row.max_t + wl*0.5) - rois.max_t)>=0
            #rois.loc[idx_min_t & idx_max_t, 'label'] = row.label

    return rois

def load_rois_annot(fname_txt, label=1):
    """ Load Raven annotations """
    rois_annot = pd.read_csv(fname_txt, delimiter='\t')
    rois_annot = rois_annot.rename(columns={'Begin Time (s)': 'min_t', 'End Time (s)': 'max_t', 
                                            'Low Freq (Hz)': 'min_f', 'High Freq (Hz)': 'max_f'})
    if not(rois_annot.empty):
        rois_annot.loc[:,'label'] = label

    return rois_annot

def load_rois_windowed(fname_annot, wl, step, tlims, flims, tn, fn):
    """ load annotations if provided """
    # load annotations if provided
    if fname_annot==None:
        rois = rois_windowed(wl, step, tlims, flims, 
                             rois_annot=None, tn=tn, fn=fn)
        rois_annot = None
    else:
        rois_annot = load_rois_annot(fname_annot)
        rois_annot = util.format_features(rois_annot, tn, fn)
        rois = rois_windowed(wl, step, tlims, flims, rois_annot, tn, fn)
    return rois, rois_annot

def load_audio(fname_audio, target_fs):
    """ Load and resample audio to target fs if necessary """
    s, fs = sound.load(fname_audio)
    if fs != target_fs:
        s = sound.resample(s, fs, target_fs, res_type='kaiser_fast')
        fs = target_fs
    return s, fs

def transform(s, fs, wtype='hann', nperseg=512, noverlap=0, db_range=60, db_gain=30):
    """ Compute decibel spectrogram """
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, wtype, nperseg, noverlap)            
    Sxx = util.power2dB(Sxx, db_range, db_gain)
    return Sxx, tn, fn, ext

def preprocess_audio_annot(fname_audio, fname_annot, opt):
    """ Define preprocessing pipeline """
    # unpack opt settings
    target_fs = opt['target_fs']
    nperseg = opt['nperseg']
    noverlap = opt['noverlap']
    wtype = opt['wtype']
    db_range = opt['db_range']
    db_gain = opt['db_gain']
    wl = opt['wl']
    step = opt['step']
    flims = opt['flims']
    tlims = opt['tlims']
    resolution = opt['resolution']
    
    # load audio and compute 2D representation
    s, fs = load_audio(fname_audio, target_fs)
    Sxx, tn, fn, ext = transform(s, fs, wtype, nperseg, noverlap, db_range, db_gain)
    
    # load annotations if provided
    rois, rois_annot = load_rois_windowed(fname_annot, wl, step, tlims, flims, tn, fn)
    
    # compute shape features
    shape, params = features.shape_features(Sxx, resolution, rois=rois)
    
    # set output
    output = {'s': s,
              'Sxx': Sxx,
              'ext': ext,
              'rois': rois,
              'rois_annot': rois_annot,
              'shape': shape}
    
    return output
 
def batch_compute_features(flist, path_audio, path_annot=None, opt=None):
    """ Compute preprocessing in batches given a list of files """
    shape_out = pd.DataFrame()
    for idx, fname in enumerate(flist):
        print(idx+1, '/', len(flist), ':',fname)
        fname_audio = path_audio+fname
        if path_annot is not None:
            fname_annot = path_annot+fname[0:-3]+'txt'
        else:
            fname_annot=None
        
        output = preprocess_audio_annot(fname_audio, fname_annot, opt)
        shape = output['shape']
        shape.loc[:,'fname'] = fname
        shape_out = shape_out.append(shape)
    
    shape_out.reset_index(inplace=True, drop=True)
    return shape_out