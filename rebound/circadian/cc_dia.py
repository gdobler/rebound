#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import utils
import cPickle as pickle
import time
import datetime
import pylab as pl
from dateutil import tz

# ---> GLOBAL VARIABLES
NUM_OBS = 2700
# TSTEPS = 10.0

DPATH = os.path.join(os.environ['REBOUND_DATA'], 'bb', '2017')
ON_STATES = os.path.join(os.environ['REBOUND_WRITE'],'circadian','light_states_tstamps_tuple.pkl')
gow_row = (900, 1200)
gow_col = (1400, 2200)
LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))[
    gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]
GOW_SRCS = np.unique(LABELS)[1:]



def load_states(spath=ON_STATES):
    '''
    spath : path to pickle object of broadband state array and timestamps
    '''
    with open(spath, 'rb') as i:
        states, bb_tstamps = pickle.load(i)

    return states, bb_tstamps

def calc_rgb(states, step=100):
    '''
    add docs!
    '''



