#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def hyper_viz(cube, img, asp=0.45):
    """
    Visualize a hyperspectral data cube.
    """

    def update_spec(event):
        if event.inaxes == axim:
            rind = int(event.ydata)
            cind = int(event.xdata)

            tspec = cube.data[:, rind, cind]
            linsp.set_data(cube.waves, cube.data[:, rind, cind])
            axsp.set_ylim(tspec.min(), tspec.max() * 1.1)
            axsp.set_title("({0},{1})".format(rind, cind))

            fig.canvas.draw()


    # -- set up the plot
    fig, ax    = plt.subplots(2, 1, figsize=(10, 10))
    axsp, axim = ax

    # -- show the image
    axim.axis("off")
    im = axim.imshow(img, "gist_gray", interpolation="nearest", aspect=asp)

    # -- show the spectrum
    axsp.set_xlim(cube.waves[0], cube.waves[-1])
    linsp, = axsp.plot(cube.waves, cube.data[:, 0, 0])

    fig.canvas.draw()
    fig.canvas.mpl_connect("motion_notify_event", update_spec)

    plt.show()

    return
