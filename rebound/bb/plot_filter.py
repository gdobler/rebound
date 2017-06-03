#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_image(img, mask=None, clim=None, oname=None):
    """
    Visualize a broadband image.
    Mask for pixel correlations.
    """

    # -- check for filename input (need to add read_raw to a utils file)
    if type(img) is str:
        img = np.fromfile(img, dtype=np.uint8).reshape(3072, 4096)

    # -- set up the plot
    xs = 5.0
    ys = xs * float(img.shape[0]) / float(img.shape[1])
    fig, ax = plt.subplots(2,1,figsize=(xs, ys))
    fig.subplots_adjust(0, 0, 1, 1)
    axr,axf = ax
    axr.axis("off")
    axf.axis("off")

    # -- create filter with alpha channel masked to pixel correlations
    filt_img = np.zeros((img.shape[0]-1,img.shape[1]-1),dtype=np.uint8)

    filt_img[mask] = 255


    # -- show the image
    im = axr.imshow(img, cmap='gist_gray',clim=clim)
    mask = axf.imshow(filt_img,cmap='gist_gray',clim=clim)
    fig.canvas.draw()
    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return
