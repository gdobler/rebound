#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os

# spatial features
# spectra (848 channels)
# Local Moran's I value

hsi = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','gow_stack_clip_2018.npy'))

nosrc_load = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','hsi_nosrc_labels.npy'))
src_load = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','hsi_src_labels.npy'))

nosrc = np.empty((nosrc_load.shape[0],nosrc_load.shape[1]+1))
src = np.empty((src_load.shape[0],src_load.shape[1]+1))

nosrc[:,:-1] = nosrc_load
src[:,:-1] = src_load

nosrc[:,-1] = 0
src[:,-1] = 1

y = np.concatenate([src, nosrc], axis=0).round().astype(int)

x = hsi[:,y[:,1],y[:,0]]

all_data = np.empty((x.shape[1],x.shape[0]+3))

all_data[:,:3] = y

all_data[:,3:] = x.T

np.random.shuffle(all_data)

# 75/25 split
cut = (all_data.shape[0]//4)*3

train =  all_data[:cut,:]
test = all_data[cut:,:]

# x,y coordinates
train_loc = train[:,:2]
test_loc = test[:,:2]

# target variable
train_y = train[:,2]
test_y = test[:,2]

# features
train_x = train[:,3:]
test_x = test[:,3:]

