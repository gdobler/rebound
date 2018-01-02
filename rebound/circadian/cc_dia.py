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
from scipy.ndimage import measurements as mm

# ---> GLOBAL VARIABLES
NUM_OBS = 1700
# TSTEPS = 10.0

DPATH = os.path.join(os.environ['REBOUND_DATA'], 'bb', '2017')
ON_STATES = os.path.join(os.environ['REBOUND_WRITE'],'circadian','light_states_tstamps_tuple.pkl')
gow_row = (900, 1200)
gow_col = (1400, 2200)
LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))[
    gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]

BB_LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'],'final','labels.npy'))
GOW_SRCS = np.unique(LABELS)[1:]
NIGHTS = [('07','29'),('07','30'),('08','01'),('08','02'),('08','03'),('08','04'),('08','05'),
          ('08','06'),('08','07'),('08','08'),('08','09'),('08','10'),('08','11'),('08','12'),
          ('08','13'),('08','14'),('08','15'),('08','16'),('08','17'),('08','18'),('08','19')]

def rg8_to_rgb(img):
    """
    Convert RG8 img to RGB image.  NOTE: no interpolation is performed, 
    only array slicing and depth stacking.  If the shape of the input 
    image is [Nr, Nc], the shape of the output image is [Nr/2, Nc/2, 3].

    Parameters:
    -----------
    img : ndarray
        An RG8 monochrome image.

    Returns:
    -------
    ndarray
        The RGB image.
    """

    red = img[::2, 1::2]
    grn = img[::2, ::2] # // 2 + img[1::2, 1::2] // 2
    blu = img[1::2, ::2]

    return np.dstack((red, grn, blu))

def load_states(spath=ON_STATES):
    '''
    spath : path to pickle object of broadband state array and timestamps
    '''
    with open(spath, 'rb') as i:
        states, bb_tstamps = pickle.load(i)

    return states, bb_tstamps

def calc_rgb(start=0, stop=1, step=30):
    '''
    add docs!
    '''
    time_start = time.time()
    nights = NIGHTS[start:stop]

    # mask for Gow sources
    mask = np.in1d(BB_LABELS, GOW_SRCS).reshape(BB_LABELS.shape)


    print "loading flist..."
    flist = [os.path.join(DPATH,n[0],n[1],i) for n in nights for i in sorted(os.listdir(os.path.join(DPATH, n[0], n[1])))[:NUM_OBS:step]]

    print "loading and stacking bb memmaps"
    data = np.empty((len(flist), BB_LABELS.shape[0],BB_LABELS.shape[1]),dtype=np.float64)

    epoch = 0

    for f in range(len(flist)):
        epoch += 1
        if epoch % (len(flist)/10)== 0:
            print "loading file # {} of {}".format(epoch, len(flist))

        img = (np.memmap(flist[f], mode='r', dtype=np.uint8).reshape(BB_LABELS.shape)*mask).copy().astype(np.float64)
        
        data[f,:,:] = img

    img75 = np.percentile(data, 75, axis=0).astype(np.uint8)

    rgb = rg8_to_rgb(img75)

    print "Time to run: {}".format(time.time() - time_start)

    return rgb