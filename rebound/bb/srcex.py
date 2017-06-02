#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time


def convert_lum(data_dir, skip=6, thresh=.99, usb=True, sh=None):
    '''
    Converts a series of raw images into a 2-D boolean mask array
    that indicates pixels that are highly correlated based on 
    changes in luminosity over time with their
    neighbors to the left or right, or above or below.

    Parameters:
    ___________
    data_dir = str
        full filepath of directory with images to be processed
        - requires RBG format

    skip = float (default 6)
        indicates number of images to skip when loading to aggregate in time

    thresh = float (default .99)
        threshold correlation coefficient value, below which all pixels are masked

    usb = bool (default True)
        indicates if images are from usb camera (i.e. 2013 images)
        - if true images are 2160 by 4096 and raw images are in RBG format

    sh = tuple of ints (default None)
        if not usb, this should be dimensions of raw images (nrow,ncols,rbb=3)

    Returns:
    ________
    2-d numpy array
        boolean array that expresses true for coordinates of pixels that are correlated
        with neighbors

    Note this output can be input into the script "plot_filter.py" to compare this mask image
    with an original raw image.

    '''
    if usb:
        sh=(2160,4096,3)

    start = time.time()
    imgs = np.array([np.fromfile(os.path.join(data_dir, i), dtype=np.uint8).reshape(sh[0],sh[1],sh[2]).mean(axis=-1) for i in sorted(os.listdir(data_dir))[::skip]])
    time_mean = time.time()

    print "Time to extract data and calculate mean: {}".format(time_mean - start)

    # stack into 3d array nrows, ncols, nimg
    imgs = imgs.transpose(1, 2, 0)
    imgs_m = imgs.copy()

    # -- subtract mean along nimg axis
    imgs_m -= imgs_m.mean(axis=2, keepdims=True)

    # - array of standard deviation for each pixel time series
    img_st = imgs.std(axis=2, keepdims=True)

    # divide x - x.mean by standard deviation
    # this will create nan values for zero division (pixels with
    # unchanging luminosity i.e. 0 st. dev)
    imgs = np.divide(imgs_m, img_st)

    # boolean mask and set nan values to 0 for further operations
    img_idx = np.isnan(imgs)

    imgs[img_idx] = 0

    time_mask = time.time()
    print "Time to stack, subtract mean and divide by std, create mask for nan values: {}".format(time_mask-time_mean)

    # create empty matrix for horizontal and vertical correlation
    cor_matrix = np.empty((sh[0], sh[1], 2))

    # algorithm to calculate horizontal and vertical correlation
    for i in range(0, sh[0]-1):
        for j in range(0, sh[1]-1):
            if j < sh[1]-1:
                # left-right correlation across columns
                cor_matrix[i, j][0] = np.dot(
                    imgs[i, j, :], imgs[i, j+1, :].T)/imgs.shape[2]
            if i < sh[0]-1:
                # up-down correlation across rows
                cor_matrix[i, j][1] = np.dot(
                    imgs[i, j, :], imgs[i+1, j, :].T)/imgs.shape[2]
    time_cor = time.time()
    print "Time to calculate correlation-cube: {}".format(time_cor-time_mask)

    # mask out negative correlation, and from this
    # filter to only correlations (in either dimension) above thresh

    # produces a 3d mask array that expresses True if a pixels rightward or downward correlation is above thresh
    # nrows by ncols by correlation in both directions
    cm = cor_matrix > thresh

    time_final_mask = time.time()
    final_mask = np.zeros((sh[0], sh[1]), dtype=bool)
    # assigns pixel and its neighbor as true (there's a more pythonesque way
    # to do this)
    for i in range(0, sh[0]-1):
        for j in range(0, sh[1]-1):
            if cm[i, j, 0]:
                final_mask[i, j] = True
                final_mask[i, j+1] = True
            if cm[i, j, 1]:
                final_mask[i, j] = True
                final_mask[i+1, j] = True

    stop = time.time()
    print "Time to create final image mask: {}".format(stop-time_final_mask)

    print "Total runtime: {}".format(stop-start)
    return final_mask
