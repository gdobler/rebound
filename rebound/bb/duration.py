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
    '''
    Loads ons and offs as datacubes, shape: num nights x num timesteps x num sources.
    '''
    return utils2.load_onoff(clip=True)


def calc_dur(ons, offs):
    """
    Takes as inputs, the on and off boolean cubes as output by load_edge.
    Returns a 2-d array of duration for light curves (each source for each night),
    shape (num nights x num sources)
    """
    ons *= 1.0
    offs *= -1.0

    master = ons + offs

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