#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

def get_tile_list():
    """
    Get the file name list for the 3D model tiles.
    """
    dpath  = os.path.join(os.environ["REBOUND_DATA"], "data_3d", "data_npy")
    fnames = [os.path.join(dpath, i) for i in sorted(os.listdir(dpath))]

    return fnames


def update_extrema(mm, tile):
    """
    Update the extrema of the footprint.

    Parameters
    ----------
    mm : list
        The current [[min_x, min_y], [max_x, max_y]] of the footprint.
    tile : ndarray
        A 3D model tile.

    Returns
    -------
    mm : list
        The updated [min, max] for the footprint.
    """

    min_x = min(mm[0][0], tile[:, 0].min())
    min_y = min(mm[0][1], tile[:, 1].min())
    max_x = max(mm[1][0], tile[:, 0].max())
    max_y = max(mm[1][1], tile[:, 1].max())

    return [[min_x, min_y], [max_x, max_y]]



if __name__ == "__main__":

    # -- get the tile list
    fnames = get_tile_list()

    # -- initialize the extrema [min, max]
    mm = [[1.0e30, 1.0e30], [-1.0e30, -1.0e30]]

    # -- for each tile, update the extrema
    for fname in fnames:
        mm = update_extrema(mm, np.load(fname))

    print("footprint:\n  x : {0}, {1}\n  y : {2}, {3}" \
              .format(mm[0][0], mm[1][0], mm[0][1], mm[1][1]))
