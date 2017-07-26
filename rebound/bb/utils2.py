#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import plotting
import matplotlib.pyplot as plt

# global variables

CURVES_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'lightcurves')
EDGE_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'final_onoff')
LABELED_MASK = np.load(os.path.join(
    os.environ['REBOUND_WRITE'], 'final', 'labels.npy'))
HSI_MASK = np.load(os.path.join(
    os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))
df = pd.read_csv(os.path.join(os.environ['REBOUND_WRITE'], 'types.csv'))
SPECTRA_CLASS = df[df[df.columns[-1]] >= 0.4]
IMG_SHAPE = (3072, 4096)
CURVE_LENGTH = 2600
NUM_CURVES = len([name for name in os.listdir(CURVES_PATH)
                  if os.path.isfile(os.path.join(CURVES_PATH, name))])
NUM_EDGES = int(len([name for name in os.listdir(
    EDGE_PATH) if os.path.isfile(os.path.join(EDGE_PATH, name))]) * 1.0 // 2)
LABELS, SIZES = np.unique(LABELED_MASK, return_counts=True)


def clip_labels(cliptype='hsi', ltype='all', lower_thresh=20, upper_sig=2):

    if cliptype == 'hsi_mask':  # all sources in HSI mask
        hsi_l, hsi_s = np.unique(HSI_MASK, return_counts=True)

        return hsi_l

    elif cliptype == 'hsi':  # sources with classified spectra
        if ltype == 'all':
            return SPECTRA_CLASS[SPECTRA_CLASS.columns[-4]].values
        else:
            selector_msk = SPECTRA_CLASS[SPECTRA_CLASS.columns[-2]] == ltype

            return SPECTRA_CLASS[selector_msk][SPECTRA_CLASS.columns[-4]].values

    elif cliptype == 'gowanus':  # sources estimated to be in Gowanus public housing
        new_hsi = HSI_MASK[1000:1100, 1610:1900].copy()
        hsi_l, hsi_s = np.unique(new_hsi, return_counts=True)

        return hsi_l

    elif cliptype == 'manual':  # manually set lower/upper bounds
        # upper thresh
        thresh = SIZES[1:].mean() + SIZES[1:].std()*upper_sig

        size_msk = (SIZES < thresh) & (SIZES > lower_thresh)

        return LABELS[size_msk]


def load_lc(cube=True, clip=False, cliptype='hsi', ltype='all'):
    """
    Loads previously extracted lightcuves. 
    If cube = False, returns a 2-d array time series (num total multi night timesteps x num sources)
    If cube = True, it does so three-dimensionally by night (num nights x timestep/night x num sources)
    """

    # get
    if cube:
        curves = np.empty((NUM_CURVES, CURVE_LENGTH, len(LABELS[1:])))

        nidx = 0

        for i in sorted(os.listdir(CURVES_PATH)):
            curves[nidx, :, :] = (np.load(os.path.join(CURVES_PATH, i)))
            nidx += 1

    else:
        all_curves = []

        for i in sorted(os.listdir(CURVES_PATH)):
            all_curves.append(np.load(os.path.join(CURVES_PATH, i)))

        curves = np.concatenate(all_curves, axis=0)

    if clip:
        idx = clip_labels(cliptype=cliptype, ltype=ltype)

        if cube:
            return curves[:, :, idx]
        else:
            return curves[:, idx]

    else:
        return curves


def load_onoff(cube=True, clip=False, cliptype='hsi', ltype='all'):

    if cube:
        ons = np.empty((NUM_EDGES, CURVE_LENGTH, len(LABELS[1:])))
        offs = np.empty((NUM_EDGES, CURVE_LENGTH, len(LABELS[1:])))

        nidx = 0
        fidx = 0

        for i in sorted(os.listdir(EDGE_PATH)):
            if i.split('_')[-1] == 'ons.npy':
                ons[nidx, :, :] = (np.load(os.path.join(EDGE_PATH, i)))
                nidx += 1

            elif i.split('_')[-1] == 'offs.npy':
                offs[fidx, :, :] = (np.load(os.path.join(EDGE_PATH, i)))
                fidx += 1

    else:
        all_ons = []
        all_offs = []

        for i in sorted(os.listdir(EDGE_PATH)):
            if i.split('_')[-1] == 'ons.npy':
                all_ons.append(np.load(os.path.join(EDGE_PATH, i)))

            elif i.split('_')[-1] == 'offs.npy':
                all_offs.append(np.load(os.path.join(EDGE_PATH, i)))

        ons = np.concatenate(all_ons, axis=0)

        offs = np.concatenate(all_offs, axis=0)

    if clip:
        idx = clip_labels(cliptype=cliptype, ltype=ltype)

        if cube:
            return ons[:, :, idx], offs[:, :, idx]
        else:
            return ons[:, idx], offs[:, idx]

    else:
        return ons, offs


def plot(data=LABELED_MASK, clip=False, cliptype='hsi', ltype='all'):
    if clip:
        final_msk = np.isin(data, clip_labels(cliptype=cliptype, ltype=ltype))

        data[~final_msk] = 0

    plotting.quick_source_info(data, clim=None, oname=None)

    return


def bar_graph():
    return
