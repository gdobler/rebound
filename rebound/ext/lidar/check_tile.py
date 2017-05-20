#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_tile(data, oname, samp=1):
    """
    Generate a scatter plot of a 3D model tile.

    Parameters
    ----------
    tile : ndarray
        3D data array
    oname : string
        output name for png
    samp : int, optional
        the spatial sampling to use
    """

    # -- set the colors
    scl  = 1e-6
    norm = data[:, 2] / data[:, 2].max() # convert height to 0 to 1
    clrs = cm.viridis(norm)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data[::samp, 0] * scl, data[::samp, 1] * scl, c=clrs[::samp], 
               lw=0)
    ax.set_xlim(data[:, 0].min()*scl, data[:, 0].max() * scl)
    ax.set_ylim(data[:, 1].min()*scl, data[:, 1].max() * scl)
    fig.canvas.draw()
    fig.savefig(oname, clobber=True)

    return


if __name__ == "__main__":

    # -- get file names
    dpath  = os.path.join(os.environ["REBOUND_DATA"], "data_3d", "data_npy")
    fnames = [os.path.join(dpath, i) for i in sorted(os.listdir(dpath))]

    # -- load the first file
    print("reading data...")
    data = np.load(fnames[0])

    # -- make a plot and write to file
    print("writing png...")
    oname = os.path.join(os.environ["REBOUND_WRITE"], "check_tile_000.png")
    plot_tile(data, oname, samp=10)
