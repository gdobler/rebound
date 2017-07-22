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

def merge(ons, offs):
    ons *= 1.0
    offs *= -1.0

    master = ons + offs

    # ---- temporary pick a night and source

    NIGHT = 0
    SOURCE = 725

    temp = master[NIGHT,:,:]

    # calculate duration
    dur = 0
    last_idx = 0
    last_on = False

    for i in range(temp.shape[0]):
        if temp[i] == 1:
            if not last_on:
                last_on = True
                last_idx = i

        elif  temp[i] == -1:
            dur += (i - last_idx)
            on = False
            last_idx = i

    return dur