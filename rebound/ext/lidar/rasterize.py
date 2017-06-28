#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import glob
import numpy as np
import pandas as pd
import laspy as lp
from utils import get_tile_list
from raster_stack import get_origin_minmax

def mn_rasterize():

    # -- set the minmax 
    mm = [[978979.241501, 194479.07369], [1003555.2415, 220149.07369]]
    
    # -- get the range
    nrow = int(round(mm[1][1] - mm[0][1] + 0.5)) + 1
    ncol = int(round(mm[1][0] - mm[0][0] + 0.5)) + 1
    
    # -- initialize the raster and counts
    rast = np.zeros((nrow, ncol), dtype=float)
    cnts = np.zeros((nrow, ncol), dtype=float)
    
    # -- set the tile names
    fnames = get_tile_list()
    nfiles = len(fnames)
    
    # -- read the tiles
    for ii, fname in enumerate(fnames):
        print("\rworking on tile {0:3} of {1}".format(ii+1, nfiles)),
        sys.stdout.flush()
        tile = np.load(fname)
    
        # -- snap to grid
        rind = (tile[:, 1] - mm[0][1]).round().astype(int)
        cind = (tile[:, 0] - mm[0][0]).round().astype(int)
    
        # -- get the counts in each bin
        tcnt = np.bincount(cind + rind * ncol)
    
        # -- update the raster
        rast[rind, cind] += tile[:, 2]
        cnts[rind, cind] += tcnt[tcnt > 0]
    
    # -- convert sum to mean
    rast /= cnts + (cnts == 0)
    
    # -- write to output
    params = {"nrow" : nrow, "ncol" : ncol, "rmin" : mm[0][1], 
              "rmax" : mm[1][1], "cmin" : mm[0][0], "cmax" : mm[1][0]}
    oname  = os.path.join(os.environ["REBOUND_WRITE"], "rasters", 
                          "MN_raster.bin")
    rast.tofile(oname)
    write_header(oname, params)


def get_lidar_tiles(location="1MT"):
    """
    Determine which LiDAR tiles overlap the region.

    Returns
    -------
    coords : dataframe
        A data frame containing tile coordinates and locations.
    """

    # -- check the location
    if location != "1MT":
        print("ONLY 1 METROTECH BK FACING WRITTEN!!!")
        return

    # -- read the list of lidar coords
    cfile  = os.path.join(os.environ["REBOUND_WRITE"], "lidar_tile_info.csv")
    coords = pd.read_csv(cfile)

    # -- get the list of tile files for BK
    fpattern = os.path.join(os.environ['REBOUND_DATA'], "data_3d")
    fpath    = "%s/*/*.TIF" % (fpattern)
    mm       = get_origin_minmax(sorted(glob.glob(fpath)))

    # -- sub-select tiles in this region
    ind = (coords.xmin >= mm[0]) & (coords.xmax < mm[2]) & \
        (coords.ymin >= mm[1]) & (coords.ymax < mm[3])

    return coords[ind]


def rasterize_lidar():
    """
    Rasterize the LiDAR tiles.
    """

    # -- get the lidar tile names and range
    coords = get_lidar_tiles()
    flist  = coords.filename.values
    nfiles = len(flist)
    mm     = [[coords.xmin.min(), coords.ymin.min()], 
              [coords.xmax.max(), coords.ymax.max()]]
    
    # -- get the range
    nrow = int(round(mm[1][1] - mm[0][1] + 0.5)) + 1
    ncol = int(round(mm[1][0] - mm[0][0] + 0.5)) + 1
    npix = nrow * ncol
    
    # -- initialize the raster and counts
    rast = np.zeros(npix, dtype=float)
    cnts = np.zeros(npix, dtype=float)
    
    # -- read the tiles
    for ii, fname in enumerate(flist):
        print("\rworking on tile {0:3} of {1}".format(ii + 1, nfiles)),
        sys.stdout.flush()
        las  = lp.file.File(fname, mode="r")
        tile = np.vstack((las.x, las.y, las.z)).T
 
        # -- snap to grid
        rind = (tile[:, 1] - mm[0][1]).round().astype(int)
        cind = (tile[:, 0] - mm[0][0]).round().astype(int)
        pind = cind + rind * ncol
    
        # -- get the counts in each bin
        tcnt = np.bincount(pind, minlength=npix)

        # -- update the raster
        rast[pind] += tile[:, 2]
        cnts[...]  += tcnt

        # -- close las file
        las.close()
    
    # -- convert sum to mean
    rast /= cnts + (cnts == 0)
    
    # -- write to output
    params = {"nrow" : nrow, "ncol" : ncol, "rmin" : mm[0][1], 
              "rmax" : mm[1][1], "cmin" : mm[0][0], "cmax" : mm[1][0]}
    oname  = os.path.join(os.environ["REBOUND_WRITE"], "rasters", 
                          "BK_raster.bin")
    rast.tofile(oname)
    write_header(oname, params)

    return
