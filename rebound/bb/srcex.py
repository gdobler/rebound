#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import measurements as mm

def find_night(data_dir,step=100,plot=False):
    '''
    Extract files and plots avg luminosity. 
    Upon click, prints file number (to indicate where to start loading)
    Upon release, prints file number (to indicate where to stop loading)
    Parameters:
    ___________
    data_dir = str
        full filepath of directory with images to be processed
        - requires RBG format

    step = int (default 100)
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

    for i in sorted(os.listdir(data_dir))[::step]:
        lum_list.append(np.memmap(os.path.join(data_dir, i), dtype=np.uint8, mode='r'))
        lum_means = np.array(lum_list,dtype=np.float32).mean(1)

    if not plot: # crude analytical method
        thresh = np.median(lum_means)+0.5
        file_start = np.where(lum_means<thresh)[0][0]*step+step
        file_stop = np.where(lum_means<thresh)[0][-1]*step

        return file_start,file_stop

    else: # plot method

        def vline_st(event):
            if event.inaxes == ax:
                cind = int(event.xdata)

                ax.axvline(cind,color='g')
                   
                ax.set_title("File #: {0}".format(cind))

                fig.canvas.draw()

                print "File start: {}".format(cind*step) + 100

        def vline_end(event):
            if event.inaxes == ax:
                cind = int(event.xdata)

                ax.axvline(cind,color='r')
                   
                ax.set_title("File #: {0}".format(cind))

                fig.canvas.draw()

                print "File stop: {}".format(cind*step)

        # -- set up the plots
        fig,ax = plt.subplots(1,figsize=(15, 5))
        im = ax.plot(lum_means)

        fig.canvas.draw()
        # fig.canvas.mpl_connect("motion_notify_event", update_spec)
        fig.canvas.mpl_connect('button_press_event', vline_st)
        fig.canvas.mpl_connect('button_release_event', vline_end)
        plt.show()


def create_mask(data_dir, step, thresh, bk, file_start, file_stop, gfilter):
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
    if file_start is None:
        print "Finding night images..."

        file_start,file_stop = find_night(data_dir)
        end_night = time.time()
        print "Time to find night: {}".format(end_night-start_mask)

    start_extract = time.time()
    print "Extracting image files..."

    if bk:
        sh = (3072, 4096)
        imgs_list = []

        for i in sorted(os.listdir(data_dir))[file_start:file_stop:step]:
            imgs_list.append(np.fromfile(os.path.join(
                    data_dir, i), dtype=np.uint8).reshape(sh[0], sh[1]))
        imgs = np.array(imgs_list,dtype=np.float32)

    else:
        sh = (2160, 4096, 3)
        imgs = np.array([np.fromfile(os.path.join(data_dir, i), dtype=np.uint8).reshape(
            sh[0], sh[1], sh[2]).mean(axis=-1) for i in sorted(os.listdir(data_dir))[::step]])


    # slice off last row and col to match mask array
    img_cube = imgs[:, :-1, :-1].copy()

    time_extract = time.time()

    print "Time to extract data and calculate mean: {}".format(time_extract - start_extract)

    # run Gaussian filter (options)
    if gfilter is not None:
        print "Running Gaussian filter..."
        imgs_sm = gf(imgs.astype(np.float64), (gfilter, 0, 0))
        imgs = imgs.astype(np.float64) - imgs_sm

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


def light_curve(data_dir, step=5, thresh=.50, bk=True, file_start=100, file_stop=2800, gfilter=None,pickle_it=None):
    '''
    Calls create_mask() and uses output to label pixels to unique light sources.
    Averages the luminosity among pixels of each light source
    to produce lightcurve for each source.
    '''
    start = time.time()

    # create mask array
    mask_array, img_cube = create_mask(
        data_dir, step, thresh, bk, file_start, file_stop, gfilter)
    
    time_label = time.time()
    # measurements.label to assign sources
    labels, num_features = mm.label(mask_array.astype(bool))

    unique, counts = np.unique(labels, return_counts=True)

    print "Creating time series array..."
    source_ts = []

    for i in range(0, img_cube.shape[0]):
        src_sum = mm.sum(img_cube[i, :, :].astype(np.float32), labels, index=unique[1:])
        source_ts.append(src_sum.astype(np.float32)/counts[1:])

    # stack sequence of time series into 2-d array time period x light source
    ts_array = np.stack(source_ts)

    time_ts_cube = time.time()
    print "Time to create time series array: {}".format(time_ts_cube - time_label)
    
    print "Broadcasting times series array to pixel image coordinates..."
    
    # broadcast timeseries of light sources into original image array (
    # NOTE: need to vectorize this

    ts_cube = np.zeros(img_cube.shape)
    for i in range(0,ts_cube.shape[1]):
        for j in range(0,ts_cube.shape[2]):
            if labels[i,j] !=0:
                ts_cube[:,i,j] = ts_array[:,labels[i,j]-1]

    ts_cube = ts_cube.astype(np.float32)
    time_output = time.time()
    print "Time to create time series cube: {}".format(time_output - time_ts_cube)

    if pickle_it != None:
        # cube
        with open(os.path.join(pickle_it,'img_cube.pickle'),'wb') as handle:
            pickle.dump(img_cube,handle,protocol=pickle.HIGHEST_PROTOCOL)

        # mask
        with open(os.path.join(pickle_it,'mask.pickle'),'wb') as handle:
            pickle.dump(mask_array,handle,protocol=pickle.HIGHEST_PROTOCOL)

        # labels
        with open(os.path.join(pickle_it,'labels.pickle'),'wb') as handle:
            pickle.dump(labels,handle,protocol=pickle.HIGHEST_PROTOCOL)

        # curves
        with open(os.path.join(pickle_it,'curves.pickle'),'wb') as handle:
            pickle.dump(ts_array,handle,protocol=pickle.HIGHEST_PROTOCOL)

        # curves_cube
        for i in np.arange(0,ts_array.shape[0]-1,10):
            with open(os.path.join(pickle_it,'curves_cube_{}.pickle'.format(i)),'wb') as handle:
                pickle.dump(ts_cube[i:i+10,:,:],handle,protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print "File loaded in output folder"
        print "Total runtime: {}".format(end - start)

    else:
        class output():

            def __init__(self):
                self.fpath = data_dir
                self.cube = img_cube
                self.mask = mask_array
                self.labels = labels
                self.unique = unique
                self.counts = counts
                self.curves = ts_array
                self.curves_cube = ts_cube

        end = time.time()
        print "Total runtime: {}".format(end - start)
        return output()




