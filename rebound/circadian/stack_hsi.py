#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import datetime
import numpy as np
import settings
import utils

def boxed_stack(input_mask, input_dir, sh, scale_factor, opath):
    '''
    Stack HSI scans truncated by a pre-set bounding box.

    Parameters
    ----------
    input_mask = 2-d numpy array
        The initial mask array with nonzero values to be bounded.

    input_dir = str
        Filepath for directory with raw HSI scans to be stacked.
    sh = tuple
        Shape of HSI scans
    scale_factor = int
        The scale by which the scan values will be reduced to prevent integer overflow when stacked.
        Should be proportional to number of initial scans. Consider 16-bit limit of 64K. 
        (e.g. 200 for raw scans by hour)
    opath = str
        Output filepath for stacked scan.

    Returns
    -------
    Saves stacked file to location at output_filepath.
    Dimensions for Gowanus mask are: ncol=286 x nwav = 848, nrow = 99
    '''
    
    rmin, rmax, cmin, cmax = utils.mask_box(input_mask)

    scan_list = []

    start = time.time()
    for i in os.listdir(input_dir):
        if i.split('.')[-1]=='raw':
        
            print('reading {}...'.format(i))

            # reads in scan, shape ncol x nwav x nrow, reverse row, reshape to nwav, nrow, ncol
            data = np.memmap(os.path.join(input_dir,i), np.uint16, mode='r').reshape(
            sh[2], sh[0], sh[1])[:,:,::-1].transpose(1,2,0)

            # truncate to bounding box dimensions and scale
            data = data[:,rmin:rmax,cmin:cmax]*1.0/scale_factor

            scan_list.append(data.astype(np.uint16))

    end_read = time.time()
    print('Time to read: {}'.format(end_read-start))
    
    print('Stacking...')
    stack = reduce(np.add, scan_list)
    end_stack = time.time()
    print('Time to stack: {}'.format(end_stack-end_read))

    stack.transpose(2, 0, 1)[..., ::-1].flatten().tofile(opath)

    return


def initial_stack():
    # -- utilities
    data_dir = settings.STACKED_HSI_FILEPATH
    opath = settings.HSI_OUT
    date_list = []
    date_hold = ''

    for i in sorted(os.listdir(data_dir))[1:]:
        fdate = i.split('_')[-2]

        if fdate == date_hold:
            data = (read_hour_stack(data_dir, i, settings.STACKED_SHAPE))*1.0/10
            date_list.append(data.astype(np.uint16))
            last = fdate
        
        else:
            if date_hold != '':
                stack_start = time.time()
                print('Time to read {}: {}'.format(last,stack_start-read_start))
                print('Stacking {} ...'.format(last))
                stack = reduce(np.add, date_list)

                stack.transpose(2, 0, 1)[..., ::-1] \
                .flatten().tofile(os.path.join(opath, last+'_stacked.raw'))
                print("Time to stack {}: {}".format(last,time.time()-stack_start))

            date_hold = fdate
            date_list = []

            print('Reading in HSI scans from {}...'.format(fdate))
            read_start = time.time()

            data = (read_hour_stack(data_dir, i, settings.STACKED_SHAPE))*1.0/10
            date_list.append(data.astype(np.uint16))

    return