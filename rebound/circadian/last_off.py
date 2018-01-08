#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
# import duration_plot
import cPickle as pickle
# import time
# import datetime
import pylab as pl
# from scipy.ndimage import measurements as mm

# ---> GLOBAL VARIABLES
DPATH = os.path.join(os.environ['REBOUND_WRITE'],'edges')
NUM_OBS = 2700
NUM_SRCS = 6870
gow_row = (900, 1200)
gow_col = (1400, 2200)
LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))[
    gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]
GOW_SRCS = np.unique(LABELS)[1:]
rb_matrix = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','rb_matrix.npy'))

def color_last_off():
    '''
    ADD DOCS!
    '''

    flist = [f for f in sorted(os.listdir(DPATH))]

    last_cube = np.empty((len(flist), NUM_SRCS))

    for f in range(len(flist)):
        with open(os.path.join(DPATH,flist[f]), 'rb') as file:
            _, _, _, last_off, ts = pickle.load(file)
        last_cube[f,:] = (last_off.T * np.arange(0,NUM_OBS)).max(axis=1)

    # get mean only for nights with light on
    masked= np.ma.masked_array(last_cube, mask=last_cube==0)

    mean_lo = np.mean(masked, axis=0).data

    return mean_lo[GOW_SRCS]

def plot_last(Y,rb_m=rb_matrix, oname=None):

    fig = pl.figure(figsize=(10,10))

    X = rb_m[:,1]

    pl.scatter(X, Y)

    pl.title("Color vs last off")
    pl.ylabel('Last off')
    pl.xlabel('"Blue-ishness"')
    pl.legend(loc='best')

    fig.canvas.draw()

    pl.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, clobber=True)

    return
# TSTEPS = 10.0