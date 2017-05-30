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


def read_raster(rname):
    """
    Read in a raster grid.
    **NOTE:** poor handling of header info at the moment...

    Parameters
    ----------
    fname : str
        The name of the raster file.

    Returns
    -------
    rast : ndarray
        The raster array.
    """

    # -- read header info
    for rec in open(rname.replace(".bin", ".hdr"), "r"):
        if "nrow" in rec:
            nrow = int(rec.split(":")[1])
        elif "ncol" in rec:
            ncol = int(rec.split(":")[1])

    # -- read in raster and reshape
    return np.fromfile(rname).reshape(nrow, ncol)
