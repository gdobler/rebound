#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import time
import datetime
from dateutil import tz

'''
Stack HSI scans during "on" times
---------------------------------


G <- READ 2-d boolean array of Gowanus sources "on" times as indexed by timestamp (nobs x nsources)

labels <- READ 2-d array of Gowanus source labels (nrows x ncols)

SEQUENCE of HSI raw files
	ht <- READ HSI timestamp

	min <- ht - 5 minutes
	max <- ht + 5 min

	FOR s in G
		s_on = False
		FOR t in G
			s_on = True if Gst is True, t > min, and t < max

		on_list <- add s_on

	final_labels <- labels in on_list : 2-d mask of pixels "on" during HSI scan

	H <- READ in raw HSI file masked by final_labels

stacked <- summation of all H in sequence along lightwave axis
'''

# first method --> determine on-state

# ---> GLOBAL VARIABLES
gow_row = (900, 1200)
gow_col = (1400, 2200)
LABELS = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final', 'hsi_pixels3.npy'))[
                 gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]
NUM_SOURCES = 6870


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

	print('Total time for {} nights: {}'.format(num_nights, time.time() - start))
	return


def precision_stack(input_dir, month, night, spath, opath, window=5):
	'''
	Input_dir : path to HSI raw files
	spath : path to pickle object of broadband state array and timestamps
	opath : path to save stacked scans
	window : temporal window within which to stack (in minutes)

	'''
	# --> utilities
	t0 = time.time()

	fpath = os.path.join(input_dir, month, night)

	with open(spath, 'rb') as i:
		states, bb_tstamp = pickle.load(i)

	sh = (848, 1600, 3194)

	HSI_list = []

	print('Time to load on-state pickle object:',time.time()-t0)

	# read in HSI
	for i in os.listdir(fpath):
		if i.split('.')[-1] == 'raw':

			print('Reading in {}...'.format(i))
			t1 = time.time()

			# read in timestamp and define window
			hsi_tstamp = int(os.path.getmtime(os.path.join(fpath,i)))

			min_bound = hsi_tstamp - window * 60
			max_bound = hsi_tstamp + window * 60

			t2 = time.time()
			print("time to read in tstamp:",t2-t1)			

			# indices of states index that are in window
			states_win = states[(bb_tstamp < max_bound) & (bb_tstamp > min_bound), :]

			t4 = time.time()

			# if any light source is on during window, evaluate to True and
			# list indices thare are the source labels in the broadband mask

			t_idx = np.any(states_win, axis=0)
			t5 = time.time()
			print('time to find index of tstamps in state array:',t5 - t4)

			# reads in scan, shape ncol x nwav x nrow, reverse row,
			# slice to Gowanus dims, transpose to nwav, nrow, ncol
			data = np.memmap(os.path.join(fpath, i), np.uint16, mode='r')

			data = data.copy().reshape(sh[2], sh[0], sh[1])[
			                 :, :, ::-1][gow_col[0]:gow_col[1], :, gow_row[0]:gow_row[1]].transpose(1, 2, 0)


			t6 = time.time()
			print('time read in HSI scan:',t6 - t5)
            
            # get 2-d mask of Gowanus sources on during window
            # ORIGINAL 2D: gow=
            # LABELS*np.in1d(LABELS,t_idx).reshape(LABELS.shape)

			mask3d = np.empty(data.shape)
			mask3d[:,:,:] = np.in1d(LABELS,np.arange(NUM_SOURCES)[t_idx]).reshape(LABELS.shape)[np.newaxis, :, :]

			t7 = time.time()
			print('time to create 3d mask:',t7 - t6)

			data[~mask3d] = 0

			t8 = time.time()
			print('time to mask data:',t8 - t7)

			HSI_list.append(data*1.0)
			print('Time for {}:{}'.format(i, time.time()-t1))

	t9 = time.time()
	print('Stacking...')
	stack = reduce(np.add, HSI_list)

	t10 = time.time()
	print("time to stack:",t10-t9)

	stack.transpose(2, 0, 1)[..., ::-1].flatten().tofile(opath)
	print("time to transpose",time.time()-t10)
	print('Total runtime {}'.format(time.time()-t0))

