#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import time
import datetime
import pylab as pl

# ---> GLOBAL VARIABLES
NUM_OBS = 2700
TSTEPS = 10.0
ON_STATES = os.path.join(os.environ['REBOUND_WRITE'],'circadian','light_states_tstamps_tuple.pkl')
gow_row = (900, 1200)
gow_col = (1400, 2200)
LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))[
    gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]
GOW_SRCS = np.unique(LABELS)[1:]
RGB_MATRIX = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','rgb_matrix_gow.npy'))

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

def plot_dur(data, sort_day, n_thresh=0.05, cm='hot', oname=None):
    '''
    Parameters:
    ----------
    Takes as input duration matrix (i.e. output of calc_dur()) and integer representing day (0-N for N nights) to sort on
    '''
    # utilities

    if data=='preload':
        data = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','duration_plot.npy'))

    # dates for ticks
    tks = np.arange(0,data.shape[0],10)
    dates = pd.date_range('06-25-2017',periods=data.shape[0], freq='D')


    if type(sort_day) == int:
        ind = np.argsort(data[sort_day, :])
        data = data[:,ind[::-1]]
        title = '"On" time duration per night for Gowanus light sources sorted on: {}'.format(dates[sort_day].date())

    elif sort_day == 'rb':
        r_b_matrix = RGB_MATRIX[:,0] - RGB_MATRIX[:,2]
        ind = np.argsort(r_b_matrix)
        data = data[:,ind[::-1]]
        title = '"On" time duration per night for Gowanus light sources sorted on red-blue intensity'

    elif sort_day == 'br':
        r_b_matrix = RGB_MATRIX[:,2] - RGB_MATRIX[:,0]
        ind = np.argsort(r_b_matrix)
        data = data[:,ind[::-1]]
        title = '"On" time duration per night for Gowanus light sources sorted on blue-red intensity'

    elif sort_day == 'bg':
        r_b_matrix = RGB_MATRIX[:,2] - RGB_MATRIX[:,1]
        ind = np.argsort(r_b_matrix)
        data = data[:,ind[::-1]]
        title = '"On" time duration per night for Gowanus light sources sorted on blue-green intensity'

    elif sort_day == 'rg':
        r_b_matrix = RGB_MATRIX[:,0] - RGB_MATRIX[:,1]
        ind = np.argsort(r_b_matrix)
        data = data[:,ind[::-1]]
        title = '"On" time duration per night for Gowanus light sources sorted on red-green intensity'

    else:
        cols = ['red', 'green','blue']
        ind = np.argsort(RGB_MATRIX[:,cols.index(sort_day)])
        data = data[:,ind[::-1]]
        title = '"On" time duration per night for Gowanus light sources sorted on: {} intensity'.format(sort_day)

    if n_thresh is not None:
        thresh = data.mean(axis=0) + 2*data.std(axis=0)
        above_thresh = data > thresh
        n_idx = (above_thresh.sum(axis=1)*1.0 / above_thresh.shape[1]) < n_thresh
        data = data[n_idx,:]

        tks = np.arange(0,data.shape[0],10)
        dates = dates[n_idx]

    fig = pl.figure(figsize=(10,10))

    pl.imshow(data.T, cmap=cm, aspect='auto')

    pl.title(title)
    pl.ylabel('Light sources')
    pl.xticks(tks,[d.date() for d in dates[tks]])
    pl.xlabel('Observation night')

    fig.canvas.draw()

    pl.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, clobber=True)

    return