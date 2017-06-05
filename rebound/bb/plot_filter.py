#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_image(mask, img=None, clim=None, oname=None):
    """
    Plot mask arrays.
    """

    # -- check for filename input (need to add read_raw to a utils file)
    if type(img) is str:
        img = np.fromfile(img, dtype=np.uint8).reshape(3072, 4096)

    # -- set up the plot
    xs = 5.0
    ys = xs * float(mask.shape[0]) / float(mask.shape[1])
    fig = plt.figure(figsize=(xs, ys))
    # fig.subplots_adjust(0, 0, 1, 1)
    # ax1,ax2 = ax
    # ax1.axis("off")
    # ax2.axis("off")

    # -- plot mask
    filt_img = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
    filt_img[mask] = 255


    # -- show the image
    # im = axr.imshow(img, cmap='gist_gray',clim=clim)
    mask = plt.imshow(filt_img,cmap='gist_gray',clim=clim)
    fig.canvas.draw()
    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return
