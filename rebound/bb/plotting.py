#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_image(img, clim=None, oname=None):
    """
    Visualize a broadband image.
    """

    # -- check for filename input (need to add read_raw to a utils file)
    if type(img) is str:
        img = np.fromfile(img, dtype=np.uint8).reshape(3072, 4096)

    # -- set up the plot
    xs = 5.0
    ys = xs * float(img.shape[0]) / float(img.shape[1])
    fig, ax = plt.subplots(figsize=(xs, ys))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    # -- show the image
    im = ax.imshow(img, cmap='gist_gray', clim=clim)
    fig.canvas.draw()
    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, clobber=True)

    return

def quick_source_info(labeled_mask, clim=None, oname=None):
    """
    Plots a mask and upon mouse hover, sets plot title to: 
        pixel row,col; 
        light source label;
        light source size.
    """

    # utils
    unique, size = np.unique(labeled_mask, return_counts=True)

    def print_label(event):
        if event.inaxes == ax:
            rind = int(event.ydata)
            cind = int(event.xdata)

            ax.set_title("Pixel: ({},{}) | Light Source #: {} | Source Pixel Size: {}".format(
                rind, cind,labeled_mask[rind,cind],size[labeled_mask[rind,cind]]))

            fig.canvas.draw()

    # -- plot mask
    mask = np.zeros(labeled_mask.shape,dtype=bool)
    idx = labeled_mask > 0
    mask[idx] = 255

    # -- set up the plot
    fig, ax= plt.subplots(1, 1, figsize=(15, 15))

    # -- show the image
    ax.axis("off")
    im = ax.imshow(mask,interpolation='nearest', cmap='gist_gray',clim=clim)

    fig.canvas.draw()
    fig.canvas.mpl_connect('motion_notify_event', print_label)

    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return

def time_series(ts_array, clim=None, oname=None):
    """
    Plots on/off transitions of light sources
    """

    # -- set up the plot
    fig = plt.figure(figsize=(10, 10))

    # -- show the image
    # axim.axis("off")
    # im = axim.imshow(mask,cmap='gist_gray',clim=clim)
    # fig.set_title('Light Curve Plot')

    # -- show the lightcurve
    # axlc.set_xlim(0, curves_obj.curves.shape[0])
    plt.plot(ts_array)

    fig.canvas.draw()

    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return

def bar_graph(data,interval):
    plt.bar()


def compare_curves(curves, ons, offs, night1, night2, off=True):

    # utils
    first_night = np.datetime64('2017-06-25')
    if off: 
        tag = -1
        tag_title = "Off"
    else:
        tag = 1
        tag_title = "On"

    # load on and off tags
    ons *= 1.0
    offs *= -1.0
    master = ons + offs

    def tags(night,sort_on):
        i,j = np.where(master[sort_on,:,:]==tag)
        _, idx = np.unique(j, return_index=True)
        
        data = curves[night]*1.0 / np.amax(curves[night],axis=0)
    
        return data[:,j[np.sort(idx)][::-1]].T

    # -- set up the plots
    fig, ax= plt.subplots(3, 1, figsize=(15, 20))
    ax1, ax2, ax3 = ax

    ax1.set_title("{} tags for {}".format(tag_title, first_night+night1))
    ax1.imshow(tags(night1,night1),cmap='gray');

    ax2.set_title("{} tags for {}, sorted on {}".format(tag_title, first_night+night2, first_night+night2))
    ax2.imshow(tags(night2, night2),cmap='gray');

    ax3.set_title("{} tags for {}, sorted on {}".format(tag_title, first_night+night2, first_night+night1))
    ax3.imshow(tags(night2, night1),cmap='gray');

    fig.canvas.draw()

    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return
