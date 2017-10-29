#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import settings
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd


def box_source(data, clim=None):
    '''
    Visually locate a source to use for registration. 
    Zoom in and press a key to print the new row and col specs.

    Parameters:
    -----------
    data : array
        The 2-d array to plot (e.g. stacked HSI scans or broadband mask)
    clim : list of ints (default None)
        Plot climit.

    Returns:
    --------
    Plots data and upon key press, prints new row and col specs.
    '''

    fig, ax = plt.subplots(1, figsize=(10, 10))

    # -- show the image
    ax.imshow(data, cmap="gist_gray", clim=clim)

    # Declare and register callbacks
    def new_limits(axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        print "*"*25
        print "row min : row max: [{:.0f}:{:.0f}]".format(math.floor(ylim[1]),math.ceil(ylim[0]))
        print "col min : col max: [{:.0f}:{:.0f}]".format(math.floor(xlim[0]),math.ceil(xlim[1]))
        print "*"*25


    fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', new_limits)

    return


def find_center(data, rmin, rmax, cmin, cmax, thresh=.8):
    '''
    Take specs from box_source() to apply ndimage processing and find center of mass.

    Parameters:
    -----------
    data : array
        2-d array used in box_source()
    rmin,rmax,cmin,cmax : int
        row and col params
    thresh : float (default .7)
        For HSI scan, threshold to isolate specific light source. Irrelevant for broadband mask.

    Returns:
    --------
    Tuple of (row,col) coordinates in original array for isolated source center of mass.
    '''

    data = data[rmin:rmax,cmin:cmax] > thresh

    label = nd.label(data)[0]

    r, c = nd.center_of_mass(data, label)

    return rmin + r, cmin + c

