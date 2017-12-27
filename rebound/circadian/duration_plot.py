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
TSTEPS = 10.0
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

def calc_dur(states):
    '''
    Takes states output array from load_states() method
    create array of nnights x nsrcs
    '''
    nnights = states.shape[0]/NUM_OBS
    nsrcs = GOW_SRCS.shape[0]

    gow_states = states[:,GOW_SRCS] # create array of desired labels

    states_time = gow_states * TSTEPS # multipy by time on for each timestep

    duration = np.empty((nnights, nsrcs), dtype=np.float64)

    for i in range(nnights):
        start = i*NUM_OBS
        end = start+NUM_OBS
        duration[i,:] = states_time[start:end, :].sum(axis=0)

    return duration

def plot_dur(data, sort_day, cm='hot'):

    if sort_day is not None:
        ind = np.argsort(data[sort_day, :])
        data = data[:,ind]
        title = 'Broadband nightly duration by night sorted on day {}'.format(sort_day)
    else:
        title = 'Broadband nightly duration by night (unsorted)'


    fig = pl.figure(figsize=(10,10))

    pl.imshow(data.T, cmap=cm, aspect='auto')

    pl.title(title)
    pl.ylabel('Sources')
    pl.xlabel('Night starting on 6/25/2017')

    # fig.canvas.draw()

    pl.show()

    return