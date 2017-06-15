#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import measurements as mm


def create_mask(data_dir, step, thresh, bk, i_start, i_stop, gfilter):
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

    step = int (default 6)
        indicates step size when iterating through files in data_dir

    thresh = float (default .50)
        threshold correlation coefficient value, below which all pixels are masked

    bk = bool (default True)
        indicates if images are from BK (i.e. May 2017) camera or (i.e. 2013 images)
        - if true images are 2160 by 4096 and raw images are in RBG format
        - if false, images are 3072 by 4096 and in monochrome8 format

    i_start = int (default None)
        index of files in source directory to start loading at 
        (remember only the half files are .raw and half .jpg-->ignored)

    i_stop = int (default None)
        index of files in source directory to stop loading

    gf = int (default None)
        if not None, implements a gaussian filter pass with sigma for time dimension set at this value


    Returns:
    ________
    2-d numpy array
        boolean array that expresses true for coordinates of pixels that are correlated
        with neighbors.
    '''
    start_mask = time.time()
    print "Extracting image files..."

    if bk:
        sh = (3072, 4096)
        imgs_list = []
        for i in sorted(os.listdir(data_dir))[i_start:i_stop:step]:
            if i.split('.')[-1] == 'raw':
                imgs_list.append(np.memmap(os.path.join(
                    data_dir, i), np.uint8, mode="r").reshape(sh[0], sh[1]))
        imgs = np.array(imgs_list, dtype='float64')

    else:
        sh = (2160, 4096, 3)
        imgs = np.array([np.fromfile(os.path.join(data_dir, i), dtype=np.uint8).reshape(
            sh[0], sh[1], sh[2]).mean(axis=-1) for i in sorted(os.listdir(data_dir))[::step]])

    # slice off last row and col to match mask array
    img_cube = imgs[:, :-1, :-1].copy()

    time_extract = time.time()

    print "Time to extract data and calculate mean: {}".format(time_extract - start_mask)

    # run Gaussian filter (options)
    if gfilter is not None:
        print "Running Gaussian filter..."
        imgs_sm = gf(1.0 * imgs, (gfilter, 0, 0))
        imgs = 1.0 * imgs - imgs_sm

    time_standard_st = time.time()
    print "Standardizing luminosity along time domain..."

    # subtract mean along nimg axis
    imgs -= imgs.mean(axis=0, keepdims=True)

    # divide by array of standard deviation for each pixel time series
    imgs /= imgs.std(axis=0, keepdims=True)

    # this will create nan values for zero division (pixels with
    # unchanging luminosity i.e. 0 st. dev)
    # boolean mask and set nan values to 0 for further operations
    img_idx = np.isnan(imgs)
    imgs[img_idx] = 0

    time_standard = time.time()
    print "Time to standardize: {}".format(time_standard-time_standard_st)
    print "Calculating correlation coefficients..."

    # matrix mult to get horizontal and vertical correlation
    corr_x = (imgs[:, :-1, :] * imgs[:, 1:, :]).mean(0)
    corr_y = (imgs[:, :, :-1] * imgs[:, :, 1:]).mean(0)

    # Creating a mask for all the pixels/sources with correlation greater than
    # threshold
    corr_mask_x = corr_x[:, :-1] > thresh
    corr_mask_y = corr_y[:-1, :] > thresh

    # Merging the correlation masks in left-right and top-down directions
    mask_array = corr_mask_x | corr_mask_y

    stop_mask = time.time()
    print "Time to create final image mask: {}".format(stop_mask-time_standard)
    print "Total create mask runtime: {}".format(stop_mask-start_mask)
    return mask_array, img_cube


def light_curve(data_dir, step=5, thresh=.50, bk=True, i_start=900, i_stop=-900, gfilter=None):
    '''
    Calls create_mask() and uses output to label pixels to unique light sources.
    Averages the luminosity among pixels of each light source
    to produce lightcurve for each source.
    '''
    start = time.time()

    # create mask array
    mask_array, img_cube = create_mask(
        data_dir, step, thresh, bk, i_start, i_stop, gfilter)

    print "Labeling light sources..."

    # measurements.label to assign sources
    labels, num_features = mm.label(mask_array.astype(bool))

    unique, counts = np.unique(labels, return_counts=True)

    # labels,unique,counts = agg_src(mask)
    time_label = time.time()
    print "Time to label: {}".format(time_label - start)
    print "Aggregate light source pixel intensities and create time series array..."
    source_ts = []

    for i in range(0, cube.shape[0]-1):
        src_sum = mm.sum(cube[i, :, :]*1.0, labels, index=unique[1:])
        source_ts.append(src_sum*1.0/counts[1:])
        # source_ts.append(src_sum*1.0)

    time_output = time.time()
    print "Time to create time series array: {}".format(time_output - time_label)

    class output():

        def __init__(self):
            self.fpath = data_dir
            self.cube = cube
            self.mask = mask
            self.labels = labels
            self.unique = unique
            self.counts = counts
            self.curves = np.stack(source_ts)

    end = time.time()
    print "Total runtime: {}".format(end - start)
    return output()
