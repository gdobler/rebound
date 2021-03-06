#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import cPickle as pickle
import matplotlib.pyplot as plt
import srcex
import bb_settings
from datetime import datetime
from dateutil import tz
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import measurements as mm


# global variables as imported from bb_settings
# DATA_FILEPATH = os.path.join(os.environ['REBOUND_DATA'], 'bb', '2017')
# IMG_SHAPE = (3072, 4096)  # dimensions of BK raw images
# MASK = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'mask.npy'))
# LABELS = np.load(os.path.join(
#     os.environ['REBOUND_WRITE'], 'final', 'labels.npy'))


def get_curves(month, night, output_dir, file_start=0, file_stop=2700, step=1, create_ts_cube=False):
    '''    
    Averages the luminosity among pixels of each light source
    to produce lightcurve for each source.

    Parameters:
    -----------
    month : str 
        Month directory of selected night

    night : str
        Night directory of selected broadband images

    output_dir : str
        Filepath for saving lightcurves
        If output_dir = None, method will return an object with unique labels, size of labels, lightcurves
            and cube of lightcurves within image coordinates, if create_ts_cube is True

    file_start : int (default 100)
        Start index for reading in images (to truncate daylight)

    file_stop : int (default 2700)
        Stop index for reading in images (to truncate daylight)

    step : int (default 1)
        Interval for reading in images -- should be 1 when producing lightcurves

    create_ts_cube : bool (default False)
        When True, the method will broadcast the 2-d array of lightcurves into a datacube that has the light
        source coordinates in the original image (nobs x nrows x ncols)

    Returns:
    --------
    If output_dir is None:
        Retuns an object with unique source labels, size of labels, lightcurves and
            if create_ts_cube True, the lightcurves broadcast into image coordinates

    Otherwise saves 2-tuple of data in output directory:
        1. data as 2-d array of nobs x nsources (or nobs x nrows x ncols if create_ts_cube True) 
        2. timestamps (integers of naive Unix timestamp)
    '''
    t0 = time.time()

    # utilities
    lnight = len(np.arange(file_start, file_stop, step))

    img_cube = np.empty((lnight, bb_settings.IMG_SHAPE[0], bb_settings.IMG_SHAPE[1]), dtype=np.float32)

    nidx = 0
    tstep = []

    t1 = time.time()
    print "Loading night files for {}_{}...".format(month, night)

    # load raw images
    for i in sorted(os.listdir(os.path.join(bb_settings.DATA_FILEPATH, month, night)))[file_start:file_stop:step]:
        tstep.append(int(os.path.getmtime(os.path.join(bb_settings.DATA_FILEPATH, month, night, i))))


        data = np.memmap(os.path.join(bb_settings.DATA_FILEPATH, month, night, i), dtype = np.uint8, mode = 'r')

        img_cube[nidx, :, :] = data.reshape(bb_settings.IMG_SHAPE[0], bb_settings.IMG_SHAPE[1]).astype(np.float32)

        nidx += 1

    unique, size = np.unique(bb_settings.LABELS_MASK, return_counts=True)

    t2 = time.time()
    print "Time to load and reshape {}_{}: {}".format(month, night, t2 - t1)
    print "Creating time series array for {}_{}...".format(month, night)
    source_ts = []


    # does not include '0' label
    for i in range(0, img_cube.shape[0]):
        src_sum = mm.sum(img_cube[i, :, :].astype(
            np.float32), bb_settings.LABELS_MASK, index=unique[1:])
        source_ts.append(src_sum.astype(np.float32)/size[1:])


    # stack sequence of time series into 2-d array time period x light source (index = unix timestamp of raw file)
    ts_array = np.stack(source_ts)

    t3= time.time()
    print "Time to create {}_{} time series array: {}".format(month, night, t3 - t2)

    # broadcast timeseries of light sources into original image array
    if create_ts_cube:
        print "Broadcasting times series array to pixel image coordinates..."

        ts_cube = np.zeros(img_cube.shape)
        for i in range(0, ts_cube.shape[1]):
            for j in range(0, ts_cube.shape[2]):
                if LABELS[i, j] != 0:
                    ts_cube[:, i, j] = ts_array.values[:, bb_settings.LABELS_MASK[i, j]-1]

        ts_cube = ts_cube.astype(np.float32)

        t4 = time.time()
        print "Time to create time series cube: {}".format(t4 - t3)

    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create tuple (data, timestep) and save to disk
        with open(os.path.join(output_dir, 'lightcurves_and_tstamps_tuple_{}_{}.pkl'.format(month, night)),'wb') as file:
            mytuple = ts_array, np.array(tstep)
            pickle.dump(mytuple, file, pickle.HIGHEST_PROTOCOL)

        # curves_cube
        if create_ts_cube:
            with open(os.path.join(output_dir, 'lightcurves_w_imgcoords_and_tstamps_tuple_{}_{}.pkl'.format(month, night)),'wb') as file:
                mytuple = ts_cube, np.array(tstep)
                pickle.dump(mytuple, file, pickle.HIGHEST_PROTOCOL)

        t5 = time.time()
        print "Total runtime: {}".format(t5 - t0)

    else:
        class output():

            def __init__(self):
                # self.cube = img_cube
                self.unique = unique
                self.size = size
                self.curves = ts_array
                if create_ts_cube:
                    self.curves_cube = ts_cube

        t6 = time.time()
        print "Total runtime: {}".format(t6 - t0)
        return output()


def multi_nights(output_dir, step=1, all_nights=False, nights=None):

    start_all = time.time()
    if all_nights:
        for m in os.listdir(bb_settings.DATA_FILEPATH):
            for n in os.listdir(os.path.join(bb_settings.DATA_FILEPATH, m)):
                get_curves(month=m, night=n,
                           output_dir=output_dir, step=step)

    else:
        if nights is None: # not include 7/24 and 7/31 due to camera data corruption
            nights = [('06', '25'), ('06', '26'), ('06', '27'), ('06', '28'), 
            ('06', '29'), ('06', '30'), ('07', '01'), ('07', '02'), ('07', '03'), 
            ('07', '04'), ('07', '05'), ('07', '06'), ('07', '07'), ('07', '08'), 
            ('07', '09'), ('07', '10'), ('07', '11'), ('07', '12'), ('07', '13'), ('07', '14'),
            ('07','15'),('07','16'),('07','17'),('07','18'),('07','19'),('07','20'),('07','21'),
            ('07','22'),('07','23'),('07','25'),('07','26'),('07','27'),('07','28'),
            ('07','29'),('07','30'),('08','01'),('08','02'),('08','03'),('08','04'),
            ('08','05'),('08','06'),('08','07'),('08','08'),('08','09'),('08','10'),('08','11'),
            ('08','12'),('08','13'),('08','14'),('08','15'),('08','16'),('08','17'),('08','18'),('08','19')]

        for n in nights:
            get_curves(month=n[0], night=n[1],
                       output_dir=output_dir, step=step)
            
    print "Total runtime for all nights: {}".format(time.time() - start_all)
