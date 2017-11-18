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


def hyper_viz(data, asp=0.45):
    """
    Visualize a hyperspectral scan. Lower plot shows mean light intensity across all wavelengths
    for full scan. Upper plot shows light intensity by wavelength for pixel hovered over.
    
    Args:
        data      : data cube of stacked HSI scans (should be nwavs by nrows by ncols)
        			Cube should be sufficiently cleaned (i.e. data - numpy.median())
        asp       :

    
    """

    imgL = data.mean(0) # mean intensity across all wavelengths


    def update_spec(event):
        if event.inaxes == axim:
            rind = int(event.ydata)
            cind = int(event.xdata)

            tspec = data[:, rind, cind]
            linsp.set_data(np.arange(0,data.shape[0]), data[:, rind, cind])
            axsp.set_ylim(tspec.min(), tspec.max() * 1.1)
            axsp.set_title("({0},{1})".format(rind, cind))

            fig.canvas.draw()


    # -- set up the plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    axsp, axim = ax

    # -- show the image
    axim.axis("off")
    im = axim.imshow(imgL, "gist_gray", interpolation="nearest", aspect=asp)
    axim.set_title('Stacked HSI scan (mean intensity across wavelength)')

    # -- show the spectrum
    axsp.set_xlim(0, data.shape[0])
    linsp, = axsp.plot(np.arange(0,data.shape[0]) ,data[:, 0, 0])

    fig.canvas.draw()
    fig.canvas.mpl_connect("motion_notify_event", update_spec)

    plt.show()

    return
