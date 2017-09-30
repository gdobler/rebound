#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import settings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def simple_plot(img, wav, setlim=[0,1000], asp=0.45):
	'''
	Plots a single wavelenth of an HSI image from an numpy array.

	Parameters
	----------
	img : numpy memmap file (e.g. output from utils.read_hour_stack)

	wav ; int
		Wavelength to plot in [0,847]

	asp : float (defaul 0.45)
		Aspect ratio

	Returns
	-------
	2-d plot of HSI scan at selected wavelength.
	'''

	fig, ax = plt.subplots(1, 1, figsize=(10,10))
	axim = ax

	im = axim.imshow(img[wav], 'gist_gray', interpolation='nearest', clim=setlim, aspect=asp)
	axim.set_title('HSI scan at {} wavelength'.format(wav))

	fig.canvas.draw()

	plt.show()

	return