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

    # -- subtract mean along nimg axis and divide by stddev along nimg axis to scale
    imgs -= imgs.mean(2,keepdims=True)

    return imgs/imgs.std(2,keepdims=True) # need to address zero values...


# -- correlate (multiply and mean) with neighboring pixels
