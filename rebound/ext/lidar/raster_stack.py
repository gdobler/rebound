
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import gdal
import glob
import numpy as np
from gdalconst import *

def get_origin_minmax(flist):
    """
    Get the minimum and maximum of the tile origins.

    Parameters
    ----------
    flist : list
        List of tile file names (full path).

    Returns
    -------
    mm : list
        List of minima and maxima of the tile origins.
    """

    # -- loop through tiles
    print("getting tile origin minima and maxima...")
    origins = np.zeros((len(flist), 2), dtype=float)
    for ii, fname in enumerate((sorted(flist))):
        tile = gdal.Open(fname, GA_ReadOnly)
        geo  = tile.GetGeoTransform()
        origins[ii] = geo[0], geo[3]

    # -- pull off and return minima and maxima
    xmin, ymin = origins.min(0)
    xmax, ymax = origins.max(0)

    return xmin, ymin, xmax, ymax


# -- get the list of tile files
fpattern = os.path.join(os.environ['REBOUND_DATA'], "data_3d", 
                        "DA18_Raster_Files", "DA18_Raster_Files", 
                        "DA18RasterTile")
fpath = "%s*.TIF" % (fpattern)
flist = glob.glob(fpath)
nfiles = len(flist)

# -- initialize the full raster
xlo, ylo, xhi, yhi = get_origin_minmax(flist)
nx_tile = int((xhi - xlo) / 2048) + 1
ny_tile = int((yhi - ylo) / 2048) + 1
nrow    = ny_tile * 2048
ncol    = nx_tile * 2048
result  = np.zeros((nrow, ncol), dtype=float)


# -- test on a tile
fname = sorted(flist)[30]
ds = gdal.Open(fname, GA_ReadOnly)
geotransform = ds.GetGeoTransform()
tile_origin = geotransform[0], geotransform[3]
raster = ds.ReadAsArray()

rind = int(tile_origin[1] - ylo)
cind = int(tile_origin[0] - xlo)

result[rind:rind + 2048, cind:cind + 2048] = raster

origin = (0, 0)
origins = []

for ii, fname in enumerate((sorted(flist))):
    print("\rreading {0} of {1} tiles...".format(ii + 1, nfiles)),
    sys.stdout.flush()
    ds = gdal.Open(fname, GA_ReadOnly)
    geotransform = ds.GetGeoTransform()
    tile_origin = geotransform[0], geotransform[3]
#    raster = ds.ReadAsArray()
    # if ii == 0:
    #     result = raster.copy()
    # elif tile_origin[0] == origin[0] + 2048 and tile_origin[1] == origin[1]: 
    #     result = np.hstack((result, raster))
    # elif tile_origin[0] == origin[0] and tile_origin[1] == origin[1] - 2048: 
    #     result = np.vstack((result, raster))
    # else:
    #     pass
    origin = tile_origin
    origins.append(tile_origin)

#print result.shape
