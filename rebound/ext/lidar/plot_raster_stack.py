#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import gdal
import numpy as np
import matplotlib.pyplot as plt
from raster_stack import get_origin_minmax
from gdalconst import GA_ReadOnly

# -- utilities
samp = 10


# -- get the file list
fpattern = os.path.join(os.environ['REBOUND_DATA'], "data_3d")
fpath    = "%s/*/*.TIF" % (fpattern)
flist    = sorted(glob.glob(fpath))
nfiles   = len(flist)


# -- get the origins
origins = np.zeros((len(flist), 2), dtype=float)
for ii, fname in enumerate((sorted(flist))):
    tile = gdal.Open(fname, GA_ReadOnly)
    geo  = tile.GetGeoTransform()
    origins[ii] = geo[0], geo[3]


# -- pull off and return minima and maxima
xlo, ylo = origins.min(0)
xhi, yhi = origins.max(0)
nrow     = int(yhi - ylo) + 2048 + 1
ncol     = int(xhi - xlo) + 2048 + 1


# -- convert origins to rows and columns
rinds, cinds = [], []
for ii, fname in enumerate(sorted(flist)):
    print("\rtile {0:4} of {1:4}...".format(ii + 1, nfiles)),
    sys.stdout.flush()
    tile = gdal.Open(fname, GA_ReadOnly)
    geo  = tile.GetGeoTransform()
    tile_origin = geo[0], geo[3]

    rinds.append(nrow - 2048 - int(tile_origin[1] - ylo))
    cinds.append(int(tile_origin[0] - xlo))


# -- read the raster
rast = np.memmap(os.path.join(os.getenv("REBOUND_WRITE"), "rasters", 
                              "BK_raster.bin"), float, mode="r") \
                              .reshape(nrow, ncol)[::samp, ::samp]


# -- get the tile colors
da18  = [i for i in flist if "DA18" in i]
da19  = [i for i in flist if "DA19" in i]
max18 = max([int(i.split("Tile")[1].split(".")[0]) for i in da18])
max19 = max([int(i.split("Tile")[1].split(".")[0]) for i in da19])
clrs  = []
for fname in flist:
    ind = int(fname.split("Tile")[1].split(".")[0])
    if "DA18" in fname:
        clrs.append(plt.cm.autumn(float(ind)/float(max18)))
    else:
        clrs.append(plt.cm.winter(float(ind)/float(max19)))


# -- set up the rectangles
fsamp = float(samp)
rects = [plt.Rectangle((i/samp, j/fsamp), 2048/fsamp, 2048/fsamp, fill=False, 
                       color=clr)
         for i, j, clr in zip(cinds, rinds, clrs)]



# -- make the plot
plt.close()
fig, ax = plt.subplots(figsize=(8.0, 8.0*float(nrow)/ncol))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, ncol/samp)
ax.set_ylim(nrow/samp, 0)
ax.imshow(rast, clim=(0, 100))
[ax.add_patch(i) for i in rects]
fig.canvas.draw()
fig.savefig(os.path.join(os.getenv("REBOUND_WRITE"), "tile_overlay.png"), 
            clobber=True)
