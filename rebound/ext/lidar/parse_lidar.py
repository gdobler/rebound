#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import geopandas as gp
import laspy as lp


def parse_las_files():
    """
    Parse the las files to determine the min/max of each las file and 
    write to an output.
    """

    # -- get the file list
    flist = sorted(glob.glob(os.path.join(os.environ["LIDAR_LAS"], "*")))
    nfile = len(flist)

    # -- open the output file for writing
    fopen = open(os.path.join(os.environ["REBOUND_WRITE"], 
                              "las_file_bounds.csv"), "w")
    fopen.write("filename,xmin,xmax,ymin,ymax,zmin,zmax,zmean,zmed\n")

    # -- loop through las files
    for ii, fname in enumerate(flist):
        print("\rparsing file {0} of {1}".format(ii + 1, nfile)),
        sys.stdout.flush()
        las = lp.file.File(fname, mode="r")
        fopen.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                        .format(fname, las.x.min(), las.x.max(), las.y.min(), 
                                las.y.max(), las.z.min(), las.z.max(), 
                                las.z.mean(), np.median(las.z)))
        las.close()

    # -- close the file
    fopen.close()

    return


def parse_lidar_shp():
    """
    Parse the lidar shp files into a more easily readable csv.
    """

    # -- read the lidar shape file
    print("reading LiDAR tile shapefile...")
    sfile = os.path.join(os.environ["LIDAR_DATA"], "Shapefile", 
                         "NYC_Final_Tile_Layout.shp")
    data  = gp.GeoDataFrame.from_file(sfile)
    ntile = len(data)

    # -- open output csv
    ofile = os.path.join(os.environ["REBOUND_WRITE"], "lidar_tile_info.csv")
    fopen = open(ofile, "w")
    fopen.write("filename,xmin,xmax,xcen,ymin,ymax,ycen\n")

    # -- write tile info to csv
    print("writing tile info to {0}".format(ofile))
    for ii in range(ntile):
        lfile  = os.path.join(os.environ["LIDAR_LAS"], data.iloc[ii].LAS_File)
        xx, yy = np.array(data.iloc[ii].geometry.boundary.xy)
        xc, yc = np.array(data.iloc[ii].geometry.centroid.xy)
        fopen.write("{0},{1},{2},{3},{4},{5},{6}\n" \
                        .format(lfile, xx.min(), xx.max(), xc[0], yy.min(), 
                                yy.max(), yc[0]))

    # -- close the file
    fopen.close()

    return


if __name__ == "__main__":

    # -- parse LiDAR shapefile
    parse_lidar_shp()
