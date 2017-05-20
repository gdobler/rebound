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
