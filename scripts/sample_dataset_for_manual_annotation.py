#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample audio dataset for manual annotation
- select hour limits:
- select number of samples
- select samples

['RUG06', 'VEG15'] have positive samples for A. femoralis
PLG10 tiene desfaso en la hora
@author: jsulloa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% set variables
hour_limits = [6, 18]
n_samples = 100  # number of samples per site
rm_sites = ['PLG20', 'PLG07']  # few samples or hour shifts
fname = '/Volumes/lacie_macosx/Dropbox/PostDoc/iavh/Putumayo/audio_metadata/data_tables/all_sites.csv'

#%% load data
df = pd.read_csv(fname)
pd.date = pd.to_datetime(df.date)
df.sensor_name.value_counts()

#%% filter by sites and time
dfsel = df.loc[~df.sensor_name.isin(rm_sites),:] 
#dfsel = dfsel.loc[(pd.date.dt.hour >= hour_limits[0]) & (pd.date.dt.hour < hour_limits[1]),:]
dfsel = dfsel.groupby('sensor_name', group_keys=False).apply(lambda x: x.sample(n_samples))

#%% check results
dfsel.sensor_name.value_counts()
len(dfsel)
dfsel.loc[:,'fname_audio'].to_csv('/Volumes/lacie_macosx/Dropbox/PostDoc/iavh/Putumayo/anfibia/fname_random_sample.csv',
                                  index=False)


