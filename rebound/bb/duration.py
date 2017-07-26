#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import utils2
import numpy as np


def calc_dur():
    """
    Takes as inputs, the on and off boolean cubes already created as outputs of detect_onoff.py
    Their shape: (num nights x num timesteps x num sources)
    Returns a 2-d array of duration for light curves (each source for each night),
    shape (num nights x num sources)
    """

    ons, offs = utils2.load_onoff(clip=True)

    ons *= 1.0
    offs *= -1.0

    master = ons + offs

    # calculate duration
    dur = np.zeros((master.shape[0], master.shape[-1]))
    last_idx = np.zeros((master.shape[0], master.shape[-1]))
    last_on = np.zeros((master.shape[0], master.shape[-1]), dtype=bool)

    for i in range(master.shape[1]):

        # if on
        on_msk = (master[:, i, :] == 1) & (~last_on)
        try:
            last_idx[on_msk] = i
            last_on[on_msk] = True
        except ValueError:
            pass

        # if off
        off_msk = master[:, i, :] == -1
        try:
            dur[off_msk] += (i - last_idx[off_msk])
            last_on[off_msk] = False
            last_idx[off_msk] = i
        except ValueError:
            pass

    return dur
