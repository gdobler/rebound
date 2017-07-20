#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt

# global variables
# location of BK bband images
CURVES_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'lightcurves') # location of lightcurves
EDGE_PATH = os.path.join(os.environ['REBOUND_WRITE'], 'on_off') # location of lightcurves
IMG_SHAPE = (3072, 4096)  # dimensions of BK raw images

def load_lc():
	all_curves = []

	for i in sorted(os.listdir(CURVES_PATH)):
		all_curves.append(np.load(os.path.join(CURVES_PATH,i)))

	return np.concatenate(all_curves, axis = 0)


def load_onoff():
	all_ons = []
	all_offs = []

	for i in sorted(os.listdir(EDGE_PATH)):
		# if i.split('_')[-1]=='ons.npy':
		# 	all_ons.append(np.load(os.path.join(EDGE_PATH,i)))

		# elif i.split('_')[-1]=='offs.npy':
		# 	all_offs.append(np.load(os.path.join(EDGE_PATH,i)))

		if i[:3]=='ons':
			all_ons.append(np.load(os.path.join(EDGE_PATH,i)))

		elif i[:3]=='off':
			all_offs.append(np.load(os.path.join(EDGE_PATH,i)))

	on_arr = np.concatenate(all_ons, axis =0)
	del all_ons

	off_arr = np.concatenate(all_offs, axis = 0)

	return on_arr, off_arr