#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import plotting
import matplotlib.pyplot as plt

# global variables
# location of BK bband images
CURVES_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'lightcurves') # location of lightcurves
EDGE_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'final_onoff') # location of lightcurves
LABELED_MASK = np.load(os.path.join(os.environ['REBOUND_WRITE'], 'final','labels.npy'))
IMG_SHAPE = (3072, 4096)  # dimensions of BK raw images
CURVE_LENGTH = 2600
NUM_CURVES = len([name for name in os.listdir(CURVES_PATH) if os.path.isfile(os.path.join(CURVES_PATH, name))])
NUM_EDGES = int(len([name for name in os.listdir(EDGE_PATH) if os.path.isfile(os.path.join(EDGE_PATH, name))]) * 1.0 //2)
LABELS,SIZES = np.unique(LABELED_MASK,return_counts=True)

def clip_labels(lower_thresh = 20, upper_sig = 2):

	# upper thresh
	thresh = SIZES[1:].mean() + SIZES[1:].std()*upper_sig

	size_msk = (SIZES < thresh) & (SIZES > lower_thresh)

	return LABELS[size_msk]

def load_lc(cube = True, clip=False):
	"""
	Loads previously extracted lightcuves. 
	If cube = False, returns a 2-d array time series (num total multi night timesteps x num sources)
	If cube = True, it does so three-dimensionally by night (num nights x timestep/night x num sources)
	"""

	# get 
	if cube:
		curves = np.empty((NUM_CURVES,CURVE_LENGTH,len(LABELS[1:])))

		nidx = 0
		
		for i in sorted(os.listdir(CURVES_PATH)):
			curves[nidx, :, :] = (np.load(os.path.join(CURVES_PATH,i)))
			nidx += 1

	else:
		all_curves = []

		for i in sorted(os.listdir(CURVES_PATH)):
			all_curves.append(np.load(os.path.join(CURVES_PATH,i)))

		curves = np.concatenate(all_curves, axis = 0)

	if clip:
		idx = clip_labels()

		if cube:
			return curves[:,:,idx]
		else:
			return curves[:,idx]

	else:
		return curves


def load_onoff(cube=True, clip=False):

	if cube:
		ons = np.empty((NUM_EDGES, CURVE_LENGTH,len(LABELS[1:])))
		offs = np.empty((NUM_EDGES, CURVE_LENGTH,len(LABELS[1:])))

		nidx = 0
		fidx = 0 

		for i in sorted(os.listdir(EDGE_PATH)):
			if i.split('_')[-1]=='ons.npy':
				ons[nidx, :, :] = (np.load(os.path.join(EDGE_PATH,i)))
				nidx += 1 

			elif i.split('_')[-1]=='offs.npy':
				offs[fidx, :, :] = (np.load(os.path.join(EDGE_PATH,i)))
				fidx += 1

	else:
		all_ons = []
		all_offs = []

		for i in sorted(os.listdir(EDGE_PATH)):
			if i.split('_')[-1]=='ons.npy':
				all_ons.append(np.load(os.path.join(EDGE_PATH,i)))

			elif i.split('_')[-1]=='offs.npy':
				all_offs.append(np.load(os.path.join(EDGE_PATH,i)))

			# if i[:3]=='ons':
			# 	all_ons.append(np.load(os.path.join(EDGE_PATH,i)))

			# elif i[:3]=='off':
			# 	all_offs.append(np.load(os.path.join(EDGE_PATH,i)))

		ons = np.concatenate(all_ons, axis =0)

		offs = np.concatenate(all_offs, axis = 0)

	if clip:
		idx = clip_labels()

		if cube:
			return ons[:, :, idx], offs[:, :, idx]
		else:
			return ons[:,idx], offs[:,idx]

	else:
		return ons, offs


def plot(data=LABELED_MASK, clip=False):
	if clip:
		final_msk = np.isin(data, clip_labels())

		data[~final_msk] = 0
	
	plotting.quick_source_info(data, clim=None, oname=None)

	return


def bar_graph():
	return