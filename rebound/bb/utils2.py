#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import plotting
import bb_settings
import datetime
import matplotlib.pyplot as plt

# global variables

# CURVES_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'lightcurves')
# EDGE_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'final_onoff')
# LABELED_MASK = np.load(os.path.join(
#     os.environ['REBOUND_WRITE'], 'final', 'labels.npy'))
# HSI_MASK = np.load(os.path.join(
#     os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))
# df = pd.read_csv(os.path.join(os.environ['REBOUND_WRITE'], 'types.csv'))
# SPECTRA_CLASS = df[df[df.columns[-1]] >= 0.4]
# IMG_SHAPE = (3072, 4096)
# CURVE_LENGTH = 2600
# NUM_CURVES = len([name for name in os.listdir(bb_settings.CURVES_FILEPATH)
#                   if os.path.isfile(os.path.join(bb_settings.CURVES_FILEPATH, name))])
# NUM_EDGES = int(len([name for name in os.listdir(
#     bb_settings.EDGE_PATH) if os.path.isfile(os.path.join(bb_settings.EDGE_PATH, name))]) * 1.0 // 2)
# LABELS, SIZES = np.unique(bb_settings.LABELS_MASK, return_counts=True)


def get_timestamp(nights):
    """
    Nights = list of 2-tuples of (month,night): str fomrmat i.e. [("06","25")]
    """
    night_dict = {}

    for n in nights:
        data_dir = os.path.join(bb_settings.DATA_FILEPATH, n[0], n[1])
        night_dict[n] = [datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(
            data_dir, file))) for file in sorted(os.listdir(data_dir))[100:2700]]

    return night_dict


def clip_labels(cliptype='hsi', light_class='all', lower_thresh=20, upper_sig=2):

    if cliptype == 'hsi_mask':  # all sources in HSI mask
        hsi_l, hsi_s = np.unique(bb_settings.FINAL_MASK, return_counts=True)

        return hsi_l

    elif cliptype == 'hsi':  # sources with classified spectra
        if light_class == 'all':
            return bb_settings.SPECTRA_CLASS[bb_settings.SPECTRA_CLASS.columns[-4]].values
        else:
            selector_msk = bb_settings.SPECTRA_CLASS[
                bb_settings.SPECTRA_CLASS.columns[-2]] == light_class

            return bb_settings.SPECTRA_CLASS[selector_msk][bb_settings.SPECTRA_CLASS.columns[-4]].values

    elif cliptype == 'gowanus':  # sources estimated to be in Gowanus public housing
        new_hsi = bb_settings.FINAL_MASK[1000:1100, 1610:1900].copy()
        hsi_l, hsi_s = np.unique(new_hsi, return_counts=True)

        return hsi_l

    elif cliptype == 'manual':  # manually set lower/upper bounds
        # upper thresh
        thresh = bb_settings.SIZES[1:].mean() + bb_settings.SIZES[1:].std()*upper_sig

        size_msk = (bb_settings.SIZES < thresh) & (bb_settings.SIZES > lower_thresh)

        return bb_settings.LABELS[size_msk]


def load_lc(cube=True, clip=None):
    """
    Loads previously extracted lightcuves. 
    If cube = False, returns a 2-d array time series (num total multi night timesteps x num sources)
    If cube = True, it does so three-dimensionally by night (num nights x timestep/night x num sources)
    """

    # get
    if cube:
        curves = np.empty((bb_settings.NUM_CURVES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))

        nidx = 0

        for i in sorted(os.listdir(bb_settings.CURVES_FILEPATH)):
            curves[nidx, :, :] = (np.load(os.path.join(bb_settings.CURVES_FILEPATH, i)))
            nidx += 1

    else:
        all_curves = []

        for i in sorted(os.listdir(bb_settings.CURVES_FILEPATH)):
            all_curves.append(np.load(os.path.join(bb_settings.CURVES_FILEPATH, i)))

        curves = np.concatenate(all_curves, axis=0)

    if clip is not None:

        if cube:
            return curves[:, :, clip]
        else:
            return curves[:, clip]

    else:
        return curves


def load_onoff(cube=True, clip=None):

    if cube:
        ons = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        offs = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))

        nidx = 0
        fidx = 0

        for i in sorted(os.listdir(bb_settings.EDGE_PATH)):
            if i.split('_')[-1] == 'ons.npy':
                ons[nidx, :, :] = (np.load(os.path.join(bb_settings.EDGE_PATH, i)))
                nidx += 1

            elif i.split('_')[-1] == 'offs.npy':
                offs[fidx, :, :] = (np.load(os.path.join(bb_settings.EDGE_PATH, i)))
                fidx += 1

    else:
        all_ons = []
        all_offs = []

        for i in sorted(os.listdir(bb_settings.EDGE_PATH)):
            if i.split('_')[-1] == 'ons.npy':
                all_ons.append(np.load(os.path.join(bb_settings.EDGE_PATH, i)))

            elif i.split('_')[-1] == 'offs.npy':
                all_offs.append(np.load(os.path.join(bb_settings.EDGE_PATH, i)))

        ons = np.concatenate(all_ons, axis=0)

        offs = np.concatenate(all_offs, axis=0)

    if clip is not None:

        if cube:
            return ons[:, :, clip], offs[:, :, clip]
        else:
            return ons[:, clip], offs[:, clip]

    else:
        return ons, offs


def plot(data=bb_settings.LABELS_MASK, clip=None):
    if clip is not None:
        final_msk = np.isin(data, clip)

        data[~final_msk] = 0

    plotting.quick_source_info(data, clim=None, oname=None)

    return


def bar_graph():
    return