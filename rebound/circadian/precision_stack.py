#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import time

'''
PSEUDO CODE

Stack HSI scans during "on" times
---------------------------------


G <- READ 2-d boolean array of Gowanus sources "on" times as indexed by timestamp (nsource x ntimeperiods)

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

def on_state(ons, offs, tstamp=None):
	'''
	Takes on and off indices and returns a 2-d array 
	that expresses true if light is on (numsources x timestep)
	'''
	lights_on = np.zeros((ons.shape[0],ons.shape[1]), dtype=bool) # boolean of light state per source x timestep
	state = np.zeros((ons.shape[0]), dtype=bool) # current state of each light source: on/off

	# forward pass
	for j in range(ons.shape[1]):

		state = ons[:,j] | state # turn state on if "on" index true (or keep on if previously on)

		state = (state & ~offs[:,j]) # turn state off if "off" index true

		lights_on[:,j] = state 	# set light state at each timestep

	# reverse indices and reset state
	lights_on = lights_on[:,::-1]
	ons = ons[:,::-1]
	offs = offs[:,::-1]
	state = np.zeros((ons.shape[0]), dtype=bool)

	# backward pass
	for j in range(ons.shape[1]):

		state = ons[:,j] | state # turn state on if "on" index true (or keep on if previously on)

		state = (state & ~offs[:,j]) # turn state off if "off" index true

		lights_on[:,j] = lights_on[:,j] | state # set light state at each timestep, but ignore if previous True

	if tstamp is not None:
		return lights_on[:,::-1], tstamp  # return lights on array (reversed back to original)
	else:
		return lights_on[:,::-1]


def multi_night(input_dir, output_dir):
	'''
	From input of director reads in on/off indices from edge object produced
	by detect_onoff and returns a 2-D array that expresses True if a light
	source is on (nsources x ntimesteps across all nights)

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
	num_sourcs = 6870
	num_nights = len([f for f in os.listdir(input_dir)])
	all_tstamps = []

	# initialize  empty array
	light_states = np.empty((num_sources, num_nights*2600), dtype=bool)

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
		light_states[:,idx:idx+2600] = on_state(ons = edge[2], offs = edge[3])

		idx += 2600
		print('Time for {}: {}'.format(i, time.time() - start_night))

	all_tstamps = np.concatenate(all_tstamps)

	df = pd.DataFrame(light_states.T, index=all_tstamps)
	
	df.to_csv(os.path.join(output_dir, 'light_states_all_time.csv'))
	print('Total time for {} nights: {}'.format(num_nights, time.time() - start))
	return