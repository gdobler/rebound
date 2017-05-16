#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg') # set back-end
import matplotlib.pyplot as plt


def plot_image(img):
    """
    Visualize a broadband image.
    """

    # -- set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    axim = ax

    # -- show the image
    axim.axis("off")
    im = axim.imshow(img, "gist_gray")

    fig.canvas.draw()

    plt.show()

    return
