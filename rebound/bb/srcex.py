#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os


def convert_lum(data_dir):
    '''
    Takes as input full filepath for directory for Oct 2013 raw data images.
    
    Reads raw image file, reshapes into hard coded dimensions 
    for images from Oct 31 2013: 2160 by 4096 pixels in RGB format

    Finds luminosity (mean of values on RGB axis)

    Returns numpy datacube of scaled luminosity values with dimensions: nrows x ncols x # of images (about 60)

    '''
    imgs = np.array([np.fromfile(os.path.join(data_dir,i),dtype=np.uint8).reshape(2160,4096,3).mean(axis=-1) for i in sorted(os.listdir(data_dir))[::6]])

    shape = imgs.shape

    # stack into 3d array nrows, ncols, nimg
    imgs = imgs.reshape(shape[1],shape[2],shape[0])
    imgs_m = imgs.copy()

    # -- subtract mean along nimg axis
    imgs_m -= imgs_m.mean(axis=2,keepdims=True)

    # - array of standard devation for each pixel time series
    img_st = imgs.std(axis=2,keepdims=True)

    # divide x - x.mean by standard deviation
    # this will create infinite values for zero division (pixels with unchanging luminosity i.e. 0 st. dev)
    imgs = np.divide(imgs_m,img_st) 

    # boolean mask and set inf values to 0 for further operations
    img_idx = imgs == np.infty

    imgs[img_idx] = 0
    
    # create empty matrix for horizontal correlation
    cor_lr = np.empty(2160,4095)

    # algorithm to calculate horizontal correlation
    for i in range(0,2160):
        for j in range(0,4095):
            cor_lr = np.dot(imgs[i,j,:],imgs[i,j+1,:].T)/imgs.shape[2]

    return cor_lr

# note there are some nan values in cor_lr
# calculate vertical correlation
# need to identify threshold for pixels of interest
