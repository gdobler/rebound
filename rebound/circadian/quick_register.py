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


def find_center(data, rmin, rmax, cmin, cmax, sig=1):
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

    data = data[rmin:rmax,cmin:cmax]

    data = data > data.std()*sig

    label = nd.label(data)[0]

    r, c = nd.center_of_mass(data, label)

    return rmin + r, cmin + c


def scale_factor(ref_pts, target_pts):
    '''
    Approximates a scale factor for two images based on input source points 
    (i.e. the points output by finding source centers in find_center()).
    A scale factor reasonably near 1 means no scaling transformation necessary.

    Input points MUST be listed in exact order for reference and target pts 
    (i.e. ref point 1 is first in the list as is its estimated counterpart target pt 1)

    Parameters:
    -----------
    ref_pts : list of tuples
        A list of reference image points in tuple form: (row, col)

    target_pts : list of tuples
        A list of target image points (in same order as ref_pts) in tuple (row, col)

    Returns:
        Floating point that approximates the scale factor of the 2 images (target >> ref) based on inputs.
    '''

    # create row and col vectors
    row_ref = np.array(ref_pts)[:,0]
    col_ref = np.array(ref_pts)[:,1]

    row_tar = np.array(target_pts)[:,0]
    col_tar = np.array(target_pts)[:,1]

    # create distance matrix
    def dist_mat(rvect, cvect):
        return np.sqrt((rvect[:,np.newaxis]-rvect)**2 + (cvect[:, np.newaxis] - cvect)**2)

    # ratio of reference over target
    dmat = dist_mat(row_ref,col_ref) / dist_mat(row_tar, col_tar)

    # upper tria of dist matrix (not including diagonal)
    dmatu = dmat[np.triu_indices(dmat.shape[0],k=1)]

    rdif = (row_tar - row_ref).sum() / len(row_tar)
    cdif = (col_tar - col_ref).sum() / len(col_tar)

    # return dmatu.sum() / len(dmatu)
    class transformed():
        def __init__(self,  data, rdif, cdif):
            self.scale = data.sum() / len(data)
            self.roffset = rdif
            self.coffset = cdif

    return transformed(dmatu, rdif, cdif)
