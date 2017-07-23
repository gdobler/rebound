#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import utils2
import numpy as np

# global variables
# location of BK bband images
EDGES = os.path.join(os.environ['REBOUND_WRITE'], 'final_onoff') # location of lightcurves

# load on / off transitions
def load_edge():
    return utils2.load_onoff(clip=True)


# create binary

def calc_dur(ons, offs):
    ons *= 1.0
    offs *= -1.0

    master = ons + offs

    # ---- temporary pick a night and source

    NIGHT = 0
    SOURCE = 725

    # master = master[NIGHT,:,:]

    # calculate duration
    dur = np.zeros((master.shape[0],master.shape[-1]))
    last_idx = np.zeros((master.shape[0],master.shape[-1]))
    last_on = np.zeros((master.shape[0],master.shape[-1]),dtype=bool)

    for i in range(master.shape[1]):

        # if on
        on_msk = (master[:,i,:] == 1) & (~last_on)
        try:
            last_idx[on_msk] = i
            last_on[on_msk] = True
        except ValueError:
            pass

        # if off
        off_msk = master[:,i,:] == -1
        try:
            dur[off_msk] += (i - last_idx[off_msk])
            last_on[off_msk] = False
            last_idx[off_msk] = i
        except ValueError:
            pass


    return dur