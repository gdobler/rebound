#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curves(curves_obj, light_source, img=None, clim=None, oname=None):
    """
    Plot light curves
    """

    # -- set up the plot
    # xs = 5.0
    # ys = xs * float(mask.shape[0]) / float(mask.shape[1])
    # fig = plt.figure(figsize=(xs, ys))
    # fig.subplots_adjust(0, 0, 1, 1)
    # ax1,ax2 = ax
    # ax1.axis("off")
    # ax2.axis("off")

    # -- plot mask
    filt_img = np.zeros(curves_obj.labels.shape,dtype=np.uint8)
    filt_img[curves_obj.labels==light_source] = 255


    # -- show the image
    # im = axr.imshow(img, cmap='gist_gray',clim=clim)
    # mask = plt.imshow(filt_img,cmap='gist_gray',clim=clim)

    # -- set up the plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    axlc, axim = ax

    # -- show the image
    axim.axis("off")
    im = axim.imshow(filt_img,cmap='gist_gray',clim=clim)
    axim.set_title('Featured Light Source')

    # -- show the lightcurve
    # axlc.set_xlim(curves_obj.curves[:,light_source], cube.waves[-1])
    linsp, = axlc.plot(curves_obj.curves[:,light_source])

    fig.canvas.draw()

    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return
