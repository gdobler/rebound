#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import measurements as mm


# -- global variables
DATA_FILEPATH = os.path.join(os.environ['REBOUND_DATA'],'bb','2017') # locatinon of BK bband images
IMG_SHAPE = (3072,4096) # dimensions of BK raw images


def truncate_daylight(data_dir, daylight_step, plot):
    '''
    Extract files and plots avg luminosity. Assumes Brooklyn data.

    For plot method:
        Upon click, prints file number (to indicate where to start loading)
        Upon release, prints file number (to indicate where to stop loading)

    Parameters:
    ___________
   data_di = str
        sub-directory filepath of night containing raw image files
        - requires RBG format

    daylight_step = int (default 100)
        indicates step size when iterating through files in data_dir

    plot = boolean (default False)
        if True, method will plot average luminosity (of each image) over time
        to allow for visual selection of optimal file start and stop

        if False, method will determine optimal start and stop analytically


    Returns:
    ________

    file_start,file_stop

    if plot = True, draws plot
    '''
    lum_list = []

    for i in sorted(os.listdir(data_dir))[::daylight_step]:
        lum_list.append(np.memmap(os.path.join(data_dir, i), dtype=np.uint8, mode='r'))
        lum_means = np.array(lum_list,dtype=np.float32).mean(1)

    if not plot: # crude analytical method
        thresh = np.median(lum_means)+0.5
        file_start = np.where(lum_means<thresh)[0][0]*daylight_step+daylight_step
        file_stop = np.where(lum_means<thresh)[0][-1]*daylight_step

        return file_start,file_stop

    else: # plot method

        def vline_st(event):
            if event.inaxes == ax:
                cind = int(event.xdata)

                ax.axvline(cind,color='g')
                   
                ax.set_title("File #: {0}".format(cind))

                fig.canvas.draw()

                print "File start: {}".format((cind*daylight_step) + 100)

        def vline_end(event):
            if event.inaxes == ax:
                cind = int(event.xdata)

                ax.axvline(cind,color='r')
                   
                ax.set_title("File #: {0}".format(cind))

                fig.canvas.draw()

                print "File stop: {}".format(cind*daylight_step)

        # -- set up the plots
        fig,ax = plt.subplots(1,figsize=(15, 5))
        im = ax.plot(lum_means)

        fig.canvas.draw()
        # fig.canvas.mpl_connect("motion_notify_event", update_spec)
        fig.canvas.mpl_connect('button_press_event', vline_st)
        fig.canvas.mpl_connect('button_release_event', vline_end)
        plt.show()

        return

def create_mask(nights, directory=DATA_FILEPATH, output=None, sh=IMG_SHAPE, step=5, file_start=100,file_stop=2700, thresh=0.5, gfilter=None):
    '''
    Converts a series of raw images into a 2-D boolean mask array
    that indicates pixels that are highly correlated based on 
    changes in luminosity over time with their
    neighbors to the left or right, or above or below.
    Assumes Brooklyn data, i.e. raw files are 3072 by 4096 and in monochrome8 format

    Parameters:
    ___________
    DATA_FILEPATH= str
        full filepath of directory that contains subdirectoris of months

    nights = list of tuples of str ('06','25')
        sub-directory of (months,nights) containing raw image files
        - requires RBG format

    step = int (default 6)
        indicates step size when iterating through files in data_dir

    thresh = float (default .50)
        threshold correlation coefficient value, below which all pixels are masked

    file_start = int (default 100)
        index of files in source directory to start loading 

    file_stop = int (default 2700)
        index of files in source directory to stop loading

    gf = int (default None)
        if not None, implements a gaussian filter pass with sigma for time dimension set at this value


    Returns:
    ________
    mask = 2-d numpy array
        boolean array that expresses true for coordinates of pixels that are correlated
        with neighbors.

    labels = 2-d numpy array
        labels of pixels grouped in unique light sources as positioned (row,col) in original image
    '''
    start_mask = time.time()

    # - utils
    lnight = len(np.arange(file_start,file_stop,step))
    nnights = len(nights)

    # initialize image time-series datacube, 
    # with known dims if using default file_start/file_stop (i.e. night only index)
    img_cube = np.empty((nnights*lnight,sh[0],sh[1]))
    idx = 0


    # -- load files for each select night and standardize
    for night in nights:
        
        # initialize night cube
        imgs = np.empty((lnight,sh[0],sh[1]))

        data_dir = os.path.join(directory,night[0],night[1])

        # load raw files
        print('Loading images for {}...'.format(night))

        for i in sorted(os.listdir(data_dir))[file_start:file_stop:step]:
            imgs = (np.fromfile(os.path.join(
                    data_dir, i), dtype=np.uint8).reshape(sh[0], sh[1]))*1.0

        # standardize
        print('Standardizing for {}...'.format(night))

        # run Gaussian filter (options)
        if gfilter is not None:
            print "Running Gaussian filter..."
            imgs_sm = gf(imgs, (gfilter, 0, 0))
            imgs -= imgs_sm

        # subtract mean along nimg axis
        imgs -= imgs.mean(axis=0, keepdims=True)

        # divide by array of standard deviation for each pixel time series
        imgs /= imgs.std(axis=0, keepdims=True)

        # this will create nan values for zero division (pixels with
        # unchanging luminosity i.e. 0 st. dev)
        # boolean mask and set nan values to 0 for further operations
        # img_idx = np.isnan(imgs)
        imgs[np.isnan(imgs)] = 0

        img_cube[idx:idx+imgs.shape[0],:,:] = imgs

        idx += imgs.shape[0]

    time_loaded = time.time()
    print('Time to load and standardize: {}'.format(time_loaded-start_mask))

    print("Calculating correlation coefficients...")

    # matrix mult to get horizontal and vertical correlation
    corr_r = (img_cube[:, :-1, :] * img_cube[:, 1:, :]).mean(0)
    corr_c = (img_cube[:, :, :-1] * img_cube[:, :, 1:]).mean(0)

    # Creating a mask for all the pixels/sources with correlation greater than
    # threshold
    corr_mask_r = corr_r > thresh
    corr_mask_c = corr_c > thresh

    # broadcasting mask to include both original pixels in all dimensions
    # across rows
    corr_mask_ru = np.append(corr_mask_r,np.zeros((1,corr_mask_r.shape[1]),dtype=bool),axis=0)
    corr_mask_rd = np.roll(corr_mask_ru,1,axis=0)
    corr_mask_r = corr_mask_ru | corr_mask_rd

    # across cols
    corr_mask_cl = np.append(corr_mask_c,np.zeros((corr_mask_c.shape[0],1),dtype=bool),axis=1)
    corr_mask_cr = np.roll(corr_mask_cl,1,axis=1)
    corr_mask_c = corr_mask_cl | corr_mask_cr

    # Merging the correlation masks in left-right and top-down directions
    mask_array = corr_mask_r | corr_mask_c


    # Create array of labeled light sources
    labels, num_features = mm.label(mask_array.astype(bool))

    stop_mask = time.time()
    print "Time to create final image mask: {}".format(stop_mask-time_loaded)
    
    if output != None:
        print "Writing files to output..."

        if not os.path.exists(output):
            os.makedirs(output)

        # write mask
        np.save(os.path.join(output,'mask.npy'),mask_array)

        # write labels
        np.save(os.path.join(output,'labels.npy'),labels)

        end = time.time()
        print "Total runtime: {}".format(end - start_mask)

    else:
        class output():

            def __init__(self):
                self.cube = img_cube
                self.mask = mask_array

        end = time.time()
        print "Total runtime: {}".format(end - start_mask)
        return output()



