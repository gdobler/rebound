#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import duration_plot
# import utils
import cPickle as pickle
import time
import datetime
import pylab as pl
# from dateutil import tz
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
RGB_MATRIX = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','rgb_matrix_gow.npy'))

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

def calc_rgb(start=0, stop=14, step=30, gow=True):
    '''
    Parameters:
    -----------
    Start and stop index for number of nights (0 = June 25, 2017)

    Stepsize (default 30, i.e. 5 minutes)

    Returns:
    2-d array of sources in Gowanus: nsrcs x (mean reddishness, mean blueishness)
    This represents the mean reddishness (R minus G) and mean bluishness (B minus G)
    across pixels in a given source.

    '''
    time_start = time.time()
    nights = NIGHTS[start:stop]

    # mask for Gow sources
    if gow:
        mask = np.in1d(BB_LABELS, GOW_SRCS).reshape(BB_LABELS.shape)
    else:
        mask = BB_LABELS>0


    print "loading flist..."
    flist = [os.path.join(DPATH,n[0],n[1],i) for n in nights for i in sorted(os.listdir(os.path.join(DPATH, n[0], n[1])))[:NUM_OBS:step]]

    print "loading and stacking bb memmaps..."
    data = np.empty((len(flist), BB_LABELS.shape[0],BB_LABELS.shape[1]),dtype=np.float64)

    epoch = 0

    for f in range(len(flist)):
        epoch += 1
        if epoch % (len(flist)/10)== 0:
            print "loading file # {} of {}".format(epoch, len(flist))

        img = (np.memmap(flist[f], mode='r', dtype=np.uint8).reshape(BB_LABELS.shape)*mask).copy().astype(np.float64)
        
        data[f,:,:] = img

    # img75 = np.percentile(data, 50, axis=0).astype(np.uint8)

    img75 = np.median(data, axis=0).astype(np.uint8)

    rgb = rg8_to_rgb(img75) * 1.0

    # extract Gowanus labels as 2-d mask from 3-d rgb cube
    maskrgb = rg8_to_rgb(BB_LABELS*mask)[:,:,0]

    # calculate color ratios
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    RGB = R + G + B

    r = np.divide(R, RGB, out = np.zeros_like(R), where = RGB != 0)
    g = np.divide(G, RGB, out = np.zeros_like(G), where = RGB != 0)
    b = np.divide(B, RGB, out = np.zeros_like(B), where = RGB != 0)

    if gow:
        rb_matrix = np.asarray([[r[maskrgb==i].mean(), g[maskrgb==i].mean(), b[maskrgb==i].mean()] for i in GOW_SRCS])
    else:
        rb_matrix = np.asarray([[r[maskrgb==i].mean(), g[maskrgb==i].mean(), b[maskrgb==i].mean()] for i in np.unique(BB_LABELS)[1:]])

    print "Time to run: {}".format(time.time() - time_start)

    return rb_matrix

def cc_plot(rb=RGB_MATRIX, duration='gow', rg_bg=False, n_thresh=0.05, cm='hot',oname=None):
    '''
    Parameters:
    -----------
    Duration: 2-d array of given sources (i.e. gowanus --> GOW_SRCS) and their total duration per night: (nnights x nsrcs)
    If "gow", it sets variable to output of duration_plot.calc_dur() for Gowanus, otherwise requires 2-d array.


    rb: 2-d array of given sources (i.e. GOW_SRCS) and the mean reddishness and mean blueishness: (nsrcx x 2). 
    nsrcs, (nrows) must equal nsrcs (ncols) in "duration" above. Default pulls pre-calculated matrix from calc_rgb
    for nights 0 (6/25/2017) to 14 at step 30.

    Returns:
    -------
    Color-color plot (x-axis = blueishness, y-axis = reddishness, color = mean nightly duration)
    '''

    if duration == 'gow':
        data = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','duration_plot.npy'))

    if n_thresh is not None:
        thresh = data.mean(axis=0) + 2*data.std(axis=0)
        above_thresh = data > thresh
        n_idx = (above_thresh.sum(axis=1)*1.0 / above_thresh.shape[1]) < n_thresh
        data = data[n_idx,:]

    nightly_duration = np.mean(data, axis=0)

    fig = pl.figure(figsize=(10,10))

    if rg_bg:
        X = rb[:,2] - rb[:,1] # blue - green
        Y = rb[:,0] - rb[:,1] # red - green
        pl.scatter(X, Y, c=nightly_duration, s=5, cmap=cm, alpha=0.5, label='Color=Mean night duration')

        pl.title("Red-Green vs Blue-Green for mean nightly duration")
        pl.ylabel('Red-Green (arb. units')
        pl.xlabel('Blue-Green (arb. units')
        pl.legend(loc='best')

    else:
        pl.scatter(rb[:,2], rb[:,0], c=nightly_duration, s=5, cmap=cm, alpha=0.5, label='Color=Mean night duration')

        pl.title("Reddish vs Bluishness for mean nightly duration")
        pl.ylabel('Relative reddishness (arb. units)')
        pl.xlabel('Relative bluishness (arb. units)')
        pl.legend(loc='best')

    fig.canvas.draw()

    pl.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, clobber=True)

    return