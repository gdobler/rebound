#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
from scipy.ndimage.filters import gaussian_filter as gf


def convert_lum(data_dir, skip=6, thresh=.99, usb=False, sh=None, i_start=900, i_stop=-900, gfilter=None):
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

    usb = bool (default False)
        indicates if images are from usb camera (i.e. 2013 images) or new (i.e. May 2017)
        - if true images are 2160 by 4096 and raw images are in RBG format
        - if false, images are 3072 by 4096 and in monochrome8 format



    sh = tuple of ints (default None)
        if not usb, this should be dimensions of raw images (nrow,ncols,rbb=3)

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
        with neighbors

    Note this output can be input into the script "plot_filter.py" to compare this mask image
    with an original raw image.

    '''
    start = time.time()
    if usb:
        sh = (2160, 4096, 3)
        imgs = np.array([np.fromfile(os.path.join(data_dir, i), dtype=np.uint8).reshape(
            sh[0], sh[1], sh[2]).mean(axis=-1) for i in sorted(os.listdir(data_dir))[::skip]])

    else:
        sh = (3072, 4096)

        imgs_list = []
        for i in sorted(os.listdir(data_dir))[i_start:i_stop:skip]:
            if i.split('.')[-1] == 'raw':
                # imgs_list.append(np.fromfile(os.path.join(data_dir, i), dtype=np.uint8).reshape(
                #         sh[0],sh[1]))

                imgs_list.append(np.memmap(os.path.join(data_dir, i), np.uint8, mode="r").reshape(sh[0], sh[1]))

        imgs = np.array(imgs_list,dtype='float64')

    time_mean = time.time()

    print "Time to extract data and calculate mean: {}".format(time_mean - start)

    # run Gaussian filter (options)
    if gfilter is not None:
        imgs_sm = gf(1.0 * imgs, (gfilter, 0, 0))
        imgs = 1.0 * imgs - imgs_sm

    # -- subtract mean along nimg axis
    imgs -= imgs.mean(axis=0, keepdims=True)

    # - divide by array of standard deviation for each pixel time series
    imgs /= imgs.std(axis=0, keepdims=True)

    # this will create nan values for zero division (pixels with
    # unchanging luminosity i.e. 0 st. dev)
    # boolean mask and set nan values to 0 for further operations
    img_idx = np.isnan(imgs)

    imgs[img_idx] = 0

    time_mask = time.time()
    print "Time to stack, subtract mean and divide by std, create mask for nan values: {}".format(time_mask-time_mean)
    
    # create empty matrix for horizontal and vertical correlation
    # cor_matrix = np.empty((sh[0], sh[1], 2))

    corr_x = (imgs[:,:-1,:] * imgs[:,1:,:]).mean(0)
    corr_y = (imgs[:,:,:-1] * imgs[:,:,1:]).mean(0)

    # algorithm to calculate horizontal and vertical correlation
    # for i in range(0, sh[0]-1):
    #     for j in range(0, sh[1]-1):
    #         if j < sh[1]-1:
    #             # left-right correlation across columns
    #             cor_matrix[i, j][0] = np.dot(
    #                 imgs[i, j, :], imgs[i, j+1, :].T)/imgs.shape[2]
    #         if i < sh[0]-1:
    #             # up-down correlation across rows
    #             cor_matrix[i, j][1] = np.dot(
    #                 imgs[i, j, :], imgs[i+1, j, :].T)/imgs.shape[2]
    time_cor = time.time()
    print "Time to calculate correlation-cube: {}".format(time_cor-time_mask)

    # mask out negative correlation, and from this
    # filter to only correlations (in either dimension) above thresh

    # produces a 3d mask array that expresses True if a pixels rightward or downward correlation is above thresh
    # nrows by ncols by correlation in both directions
    # cm = cor_matrix > thresh

    time_final_mask = time.time()
    # final_mask = np.zeros((sh[0], sh[1]), dtype=bool)
    # assigns pixel and its neighbor as true (there's a more pythonesque way
    # to do this)
    # for i in range(0, sh[0]-1):
    #     for j in range(0, sh[1]-1):
    #         if cm[i, j, 0]:
    #             final_mask[i, j] = True
    #             final_mask[i, j+1] = True
    #         if cm[i, j, 1]:
    #             final_mask[i, j] = True
    #             final_mask[i+1, j] = True

    # Creating a Mask for all the pixels/sources with correlation greater than threshold
    corr_mask_x = corr_x[:,:-1] > thresh
    corr_mask_y = corr_y[:-1,:] > thresh

    # Merging the correlation masks in left-right and top-down directions
    final_mask = corr_mask_x | corr_mask_y


    stop = time.time()
    print "Time to create final image mask: {}".format(stop-time_final_mask)

    print "Total runtime: {}".format(stop-start)
    return final_mask


def agg_src(mask_array):
    """
    Takes as input the output of .conver_lum() -- mask array of correlated pixels.
    Groups them into contiguous groups, assigns unique labels to the groups,
    Counts number and size of groups <-> light sources.
    """

    # measurement.label to assign sources

    # measurement.sum to count sources (maybe)

    return # datacube: nrows x ncols x source label


def light_curve(source_array, cube):
    '''
    Takes as inpout the output of agg_src() and oringial datacube.
    Averages the luminosity among pixels of each light source
    to produce lightcurve for each source.

    '''

    # mask datacube with source_array to average luminosity 
    # of each source, iterating through each image in the series

    # zero out non-source pixels to create sparse cube that maintains
    # light source spatial coordinates

    return # rank 4 tensor of luminosity values: nrows x ncols x nimgs x 1 (source label)
