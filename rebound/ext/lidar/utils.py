#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
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


def write_header(fname, params):
    """
    Write a header for a binary raster file.

    Parameters
    ----------
    fname : str
        The filename of the raster (should end in ".bin").
    params : dict
        The parameters to write to the header file.
    """

    # -- open the file and write output time
    fopen = open(fname.replace(".bin", ".hdr"), "w")
    fopen.write("{0}\n".format(str(datetime.datetime.now())))

    # -- loop through parameters
    for k, v in params.items():
        fopen.write("{0} : {1}\n".format(k, str(v)))

    # -- close the file
    fopen.close()

    return


