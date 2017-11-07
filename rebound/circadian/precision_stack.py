#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

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

def on_state(ons, offs, tstamp):
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


	return lights_on[:,::-1], tstamp # return lights on array (reversed back to original) and tstamp index



