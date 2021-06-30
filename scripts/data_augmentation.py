#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesize soundcapes from real audio files using the Scaper package.

"""

import numpy as np
import scaper
import os


path_to_audio = os.path.expanduser('../data_augmentation')
# output folder
out_folder = 'synth/'
# Scaper settings
fg_folder = 'foreground/'
bg_folder = 'background/'
soundscape_duration = 60.0
n_soundscapes = 50
sampling_rate = 24000
ref_db = -50  # reference loudness for background, closer to 0=louder
min_snr, max_snr = (6,7)
seed = 123
n_channels = 1
min_events, max_events = (1,2)

output_folder = os.path.join(path_to_audio, out_folder)
foreground_folder = os.path.join(path_to_audio, fg_folder)
background_folder = os.path.join(path_to_audio, bg_folder)
sc = scaper.Scaper(soundscape_duration, foreground_folder, background_folder)
sc.ref_db = ref_db  
sc.sr = sampling_rate
sc.n_channels = n_channels


#%%
for n in range(n_soundscapes):

    print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))

    # reset the event specifications for foreground and background at the
    # beginning of each loop to clear all previously added events
    sc.reset_bg_event_spec()
    sc.reset_fg_event_spec()

    # Add background
    sc.add_background(label=('choose', ['rain', 'waterflow', 'insects']),
                      source_file=('choose', []),
                      source_time=('const', 0))

    # add random number of foreground events
    n_events = np.random.randint(min_events, max_events+1)
    for _ in range(n_events):
        sc.add_event(label=('const', 'ALFE'),
                     source_file=('choose', []),
                     source_time=('const', 0),
                     event_time=('uniform', 0, 50),
                     event_duration=('truncnorm', 8, 3, 3, 15),
                     snr=('normal', min_snr, max_snr),
                     pitch_shift=('uniform', -0.2, 0.2),
                     time_stretch=('uniform', 0.9, 1.1))
    
    # generate
    audiofile = os.path.join(output_folder, "soundscape_ALFE{:d}.wav".format(n))
    jamsfile = os.path.join(output_folder, "soundscape_ALFE{:d}.jams".format(n))
    txtfile = os.path.join(output_folder, "soundscape_ALFE{:d}.txt".format(n))
    
    sc.generate(audiofile, jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=None,
                disable_sox_warnings=True,
                no_audio=False,
                peak_normalization=True,
                txt_path=txtfile)

#%% Reformat labels
import glob
import pandas as pd
flist = glob.glob(output_folder+'*.txt')
for fname in flist:
    df_annot = pd.read_csv(fname, delimiter='\t', header=None)
    df_annot = df_annot.rename(columns={0: 'Begin Time (s)', 
                                        1: 'End Time (s)',
                                        2: 'label'})
    df_annot.loc[:, 'Low Freq (Hz)'] = 2500
    df_annot.loc[:, 'High Freq (Hz)'] = 5500
    df_annot.to_csv(fname, sep='\t', index=False)