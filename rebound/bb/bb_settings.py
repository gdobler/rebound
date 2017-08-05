#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


# topdirectory for 2017 broadband raw images
DATA_FILEPATH = os.path.join(os.environ['REBOUND_DATA'], 'bb', '2017')

# dimensions of 2017 raw images
IMG_SHAPE = (3072, 4096)

# final boolean mask
BOOL_MASK = np.load(os.path.join(
    os.environ['REBOUND_WRITE'], 'final', 'mask.npy'))

# final mask with labels
LABELS_MASK = np.load(os.path.join(
    os.environ['REBOUND_WRITE'], 'final', 'labels.npy'))

# lightcurves directory
CURVES_FILEPATH = os.path.join(os.environ['REBOUND_WRITE'], 'lightcurves')

# on and off edges directory
EDGE_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'final_onoff')

# bb-hsi merged mask
FINAL_MASK = np.load(os.path.join(
    os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))

# dataframe of spectra classes for 0.4 threshold
df = pd.read_csv(os.path.join(os.environ['REBOUND_WRITE'], 'types.csv'))
SPECTRA_CLASS = df[df[df.columns[-1]] >= 0.35]

# length of lightcurves (i.e. number of timesteps)
CURVE_LENGTH = 2600

# number of lightcurves created
NUM_CURVES = len([name for name in os.listdir(CURVES_FILEPATH)
                  if os.path.isfile(os.path.join(CURVES_FILEPATH, name))])

# number of edges created
NUM_EDGES = int(len([name for name in os.listdir(
    EDGE_PATH) if os.path.isfile(os.path.join(EDGE_PATH, name))]) * 1.0 // 2)

# unique labels, sizes
LABELS, SIZES = np.unique(LABELS_MASK, return_counts=True)