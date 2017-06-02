#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time

def convert_lum(data_dir):
    '''
    Takes as input full filepath for directory for Oct 2013 raw data images.
    
    Reads raw image file, reshapes into hard coded dimensions 
    for images from Oct 31 2013: 2160 by 4096 pixels in RGB format

    Finds luminosity (mean of values on RGB axis)

    Returns numpy datacube of scaled luminosity values with dimensions: nrows x ncols x # of images (about 60)

    '''
    start = time.time()
    imgs = np.array([np.fromfile(os.path.join(data_dir,i),dtype=np.uint8).reshape(2160,4096,3).mean(axis=-1) for i in sorted(os.listdir(data_dir))[::6]])
    time_mean = time.time()

    print "Time to extract data and calculate mean: {}".format(time_mean - start)

    # stack into 3d array nrows, ncols, nimg
    imgs = imgs.transpose(1,2,0)
    imgs_m = imgs.copy()

    # -- subtract mean along nimg axis
    imgs_m -= imgs_m.mean(axis=2,keepdims=True)

    # - array of standard devation for each pixel time series
    img_st = imgs.std(axis=2,keepdims=True)

    # divide x - x.mean by standard deviation
    # this will create infinite values for zero division (pixels with unchanging luminosity i.e. 0 st. dev)
    imgs = np.divide(imgs_m,img_st) 

    # boolean mask and set inf values to 0 for further operations
    img_idx = np.isnan(imgs)

    imgs[img_idx] = 0

    time_mask = time.time()
    print "Time to stack, subtract mean and divide by std, create mask for nan values: {}".format(time_mask-time_mean)
    
    # create empty matrix for horizontal and vertical correlation
    cor_matrix = np.empty((2160,4096,2))

    # algorithm to calculate horizontal and vertical correlation
    for i in range(0,2159):
        for j in range(0,4095):
            if j<4095:
                # left-right correlation across columns
                cor_matrix[i,j][0] = np.dot(imgs[i,j,:],imgs[i,j+1,:].T)/imgs.shape[2]
            if i<2159:
                # up-down correlation across rows
                cor_matrix[i,j][1] = np.dot(imgs[i,j,:],imgs[i+1,j,:].T)/imgs.shape[2]
    time_cor = time.time()
    print "Time to calculate correlation-cube: {}".format(time_cor-time_mask)
    
    # mask out negative correlation, and from this
    # filter to only correlations (in either dimension) above thresh
    thresh = 0.99

    # produces a 3d mask array that expresses True if a pixels rightward or downward correlation is above thresh
    # nrows by ncols by correlation in both directions (2160,4096,2)
    cm = cor_matrix > thresh
    
    time_final_mask = time.time()
    final_mask = np.zeros((2160,4096),dtype=bool)
    # assigns pixel and its neighbor as true (there's a more pythonesque way to do this)
    for i in range(0,2159):
        for j in range(0,4095):
            if cm[i,j,0]:
                final_mask[i,j] = True
                final_mask[i,j+1] = True
            if cm[i,j,1]:
                final_mask[i,j]=True
                final_mask[i+1,j]=True
    
    stop = time.time()
    print "Time to create final image mask: {}".format(stop-time_final_mask)
    
    print "Total runtime: {}".format(stop-start)
    return final_mask


