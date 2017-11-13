#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import utils
import cPickle as pickle
import time
import datetime
from dateutil import tz

# ---> GLOBAL VARIABLES
gow_row = (900, 1200)
gow_col = (1400, 2200)
LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))[
    gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]
NUM_SOURCES = 6870
NIGHTS = [('07','29'),('07','30'),('08','01'),('08','02'),('08','03'),('08','04'),('08','05'),
          ('08','06'),('08','07'),('08','08'),('08','09'),('08','10'),('08','11'),('08','12'),
          ('08','13'),('08','14'),('08','15'),('08','16'),('08','17'),('08','18'),('08','19')]

def on_state(ons, offs, tstamp=None):
    '''
    Takes on and off indices (num obs x num sources) and returns a 2-d array
    that expresses true if light is on (nobs x nsources)
    '''
    lights_on = np.zeros((ons.shape[0], ons.shape[
                         1]), dtype=bool)  # boolean of light state per obs x source
    # current state of each light source: on/off
    state = np.zeros((ons.shape[1]), dtype=bool)

    # forward pass
    for i in range(ons.shape[0]):

        # turn state on if "on" index true (or keep on if previously on)
        state = ons[i, :] | state

        state = (state & ~offs[i, :])  # turn state off if "off" index true

        lights_on[i, :] = state 	# set light state at each timestep

    # reverse indices and reset state
    lights_on = lights_on[::-1, :]
    ons = ons[::-1, :]
    offs = offs[::-1, :]
    state = np.zeros((ons.shape[1]), dtype=bool)

    # backward pass
    for i in range(ons.shape[0]):

        # turn state on if "on" index true (or keep on if previously on)
        # note that in reverse the "off" index is the "on"
        state = offs[i, :] | state

        state = (state & ~ons[i, :])  # turn state off if "off" index true

        # set light state at each timestep, but ignore if previous True
        lights_on[i, :] = lights_on[i, :] | state

    if tstamp is not None:
        # return lights on array (reversed back to original)
        return lights_on[::-1, :], tstamp
    else:
        return lights_on[::-1, :]


def multi_night(input_dir, output_dir):
    '''
    From input of director reads in on/off indices from edge object produced
    by detect_onoff and returns a 2-D array that expresses True if a light
    source is on (nobs across all nights x nsources)

    Parameters:
    -----------
    input_dir : str
            Filepath for directory with edge objects.

            Each edge object is a tuple with component at index:
            0: lightcurves
            1: lc after Gaussian filter
            2: on indices
            3: off indices
            4: timestamp (index)
    '''
    start = time.time()

    # utilities

    num_nights = len([f for f in os.listdir(input_dir)])
    all_tstamps = []

    # initialize  empty array
    light_states = np.empty((num_nights*2600, NUM_SOURCES), dtype=bool)

    idx = 0

    for i in sorted(os.listdir(input_dir)):
        start_night = time.time()
        print("Loading {}".format(i))

        with open(os.path.join(input_dir, i), 'rb') as p:
            edge = pickle.load(p)

        # append timestamp
        all_tstamps.append(edge[4])

        print('Determining on state for {}'.format(i))
        # run on_state
        light_states[idx:idx+2600, :] = on_state(ons=edge[2], offs=edge[3])

        idx += 2600
        print('Time for {}: {}'.format(i, time.time() - start_night))

    all_tstamps = np.concatenate(all_tstamps)

    with open(os.path.join(output_dir, 'light_states_tstamps_tuple.pkl'), 'wb') as file:
        l = light_states, all_tstamps
        pickle.dump(l, file, pickle.HIGHEST_PROTOCOL)

    print('Total time for {} nights: {}'.format(
        num_nights, time.time() - start))
    return


def load_states(spath):
    '''
    spath : path to pickle object of broadband state array and timestamps
    '''
    with open(spath, 'rb') as i:
        states, bb_tstamps = pickle.load(i)

    return states, bb_tstamps


def precision_stack(input_dir, month, night, states, bb_tstamps, opath, window=5):
    '''
    Input_dir : path to HSI raw files
    opath : path to save stacked scans
    window : temporal window within which to stack (in minutes)

    '''
    # --> utilities
    t0 = time.time()
    dir_path = os.path.join(input_dir, month, night)
    HSI_list = []

    # read in HSI
    for i in sorted(os.listdir(dir_path))[::5]:
        if i.split('.')[-1] == 'raw':
            fpath = os.path.join(dir_path, i)
            print('Reading in HSI {}...'.format(i))
            t1 = time.time()

            # read in timestamp and define window
            hsi_tstamp = int(os.path.getmtime(fpath))

            min_bound = hsi_tstamp - window * 60
            max_bound = hsi_tstamp + window * 60

            # indices of states index that are in window                                                                            
            states_win = states[(bb_tstamps < max_bound) & (bb_tstamps > min_bound), :]
            
            if states_win.shape[0] == 0:
                print "No broadband image during window (i.e. truncated for daylight)..."

            else:
                # -- read the header
                hdr = utils.read_header(fpath.replace("raw", "hdr"))
                sh  = (hdr["nwav"], hdr["nrow"], hdr["ncol"])

                t2 = time.time()
                print "Time to read in HSI header:",t2-t1
                print "Reading in HSI data..."

                data = np.memmap(fpath, np.uint16, mode='r')

                data = data.copy().reshape(sh[2], sh[0], sh[1])[
		   		:, :, ::-1][gow_col[0]:gow_col[1],:,gow_row[0]:gow_row[1]].transpose(1, 2, 0)

                print"Data shape is: {}".format(data.shape)
                t3 = time.time()
                print "Time to read and reshape HSI scan:", t3-t2
                print("Creating mask...")

                t_idx = np.any(states_win, axis = 0)

                mask3d = np.empty(data.shape)
                mask3d[:, :, :] = np.in1d(LABELS, np.arange(states.shape[1])[t_idx]).reshape(LABELS.shape)[np.newaxis, :, :]

                mask3d = mask3d.astype(bool)

                t4 = time.time()
                print 'Time to create 3d mask:', t4 - t3

                data[~mask3d] = 0

                t5 = time.time()
                print 'Time to mask data:', t5 - t4

                HSI_list.append(data.astype(np.float32))
                print 'Time for {}:{}'.format(i, time.time()-t1)

    print 'Stacking...'
    stack = reduce(np.add, HSI_list)

    print "Total runtime {}".format(time.time() - t0)

    return stack

def multi_stack(input_dir, spath, opath, night_list=NIGHTS, window=5):
    '''
    Runs precision stack method through list of months and nights.
    '''
    t0 = time.time()
    # load array of light states and timestamps
    print "Loading on states array and timestamps..."
    states, bb_ts = load_states(spath)

    all_hsi_list = []

    for n in night_list:
        t1 = time.time()
        print "Precision stacking for {}/{}".format(n[0],n[1])

        all_hsi_list.append(precision_stack(input_dir=input_dir, month = n[0], night = n[1], 
            states = states, bb_tstamps = bb_ts, opath=opath, window=window))
        print "Time for {}/{}: {}".format(n[0],n[1],time.time()-t1)

    t2 = time.time()

    print "Stacking all nights..."
    all_stack = reduce(np.add, all_hsi_list)
    print "Time to stack : {}".format(time.time() - t2)

    all_stack.transpose(2, 0, 1)[..., ::-1].flatten().tofile(opath)

    return
