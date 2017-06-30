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
    im = ax.imshow(img, cmap="gist_gray", clim=clim)
    fig.canvas.draw()
    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, clobber=True)

    return

def plot_curves(curves_obj, clim=None, oname=None):
    """
    Plot light curves
    """

    def update_ts(event):
        if event.inaxes == axim:
            rind = int(event.ydata)
            cind = int(event.xdata)

            tspec = curves_obj.curves[:, rind, cind]
            linsp.set_data(np.asarray(np.arange(curves_obj.curves.shape[0])),curves_obj.curves[:, rind, cind])
            axlc.set_ylim(tspec.min(), tspec.max() * 1.1)
            axlc.set_title("({0},{1})".format(rind, cind))

            fig.canvas.draw()

    # -- plot mask
    mask = np.zeros(curves_obj.mask.shape,dtype=np.uint8)
    mask[curves_obj.mask] = 255


    # -- set up the plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    axlc, axim = ax

    # -- show the image
    axim.axis("off")
    im = axim.imshow(mask,cmap='gist_gray',clim=clim)
    axim.set_title('Mask Image')

    # -- show the lightcurve
    axlc.set_xlim(0, curves_obj.curves.shape[0])
    linsp, = axlc.plot(np.asarray(np.arange(curves_obj.curves.shape[0])),curves_obj.curves[:, 0, 0])

    fig.canvas.draw()
    fig.canvas.mpl_connect("motion_notify_event", update_ts)

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