#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curves(curves_obj, light_source=None, img=None, clim=None, oname=None):
    """
    Plot light curves
    """

    def update_spec(event):
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
    fig.canvas.mpl_connect("motion_notify_event", update_spec)

    plt.show()

    # -- write to file if desired
    if oname is not None:
        fig.savefig(oname, format='png',clobber=True)

    return
