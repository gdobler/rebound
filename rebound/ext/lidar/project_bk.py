#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from utils import read_raster
from rasterize import get_lidar_tiles
from colin_inv import *


# -- load the raster
try:
    rast
except:
    fname = os.path.join(os.environ["REBOUND_WRITE"], "rasters",
                         "BK_raster.bin")
    print("reading raster {0}...".format(fname))
    rast  = read_raster(fname)


# -- set the camera parameters

# params = np.array([4.71238898038469, -1.26155483e-02, 2.39850465e-02,
#                    9.87891059e+05, 1.91728245e+05, 4.00726823e+02,
#                    1.63556904e+04])

#params = np.array([4.71238898038469, -1.26155483e-02, 2.39850465e-02,
#                    9.87891059e+05, 1.91728245e+05, 5.00726823e+02,
#                    1.63556904e+04])

#params = np.array([1.46161933 + np.pi, 2.0 * -1.26155483e-02, 0.0,
#                   9.87891059e+05, 1.91728245e+05, 5.00726823e+02,
#                   1.63556904e+04])

params = np.array([4.59086617e+00, -5.60380015e-02, 5.87195417e-03,
                   9.87932695e+05, 1.91780076e+05, 3.16128741e+02,
                   1.53237201e+04]) #best params

x0     = params[3]
y0     = params[4]

# -- initialize an image
nrow = 3072
ncol = 4096

dgrid = np.zeros((nrow, ncol), dtype=float)
xgrid = np.zeros((nrow, ncol), dtype=float)
ygrid = np.zeros((nrow, ncol), dtype=float)

# -- loop through pixels
#    - identify the x,y,z coordinates for all rs
#    - convert x,y coordinates to indices
#    - find all x,y coordinates for which z is greater than projectws line
#    - of those, find the closest

# -- get the lidar tile names and range
coords = get_lidar_tiles()
mm     = [[coords.xmin.min(), coords.ymin.min()],
          [coords.xmax.max(), coords.ymax.max()]]

rs = np.arange(50, 50000., 1.)

#for ii in range(nrow)[::10]:
#    print("\r{0} : {1}".format(ii+1, nrow)),
#    sys.stdout.flush()
#    for jj in range(ncol)[::10]:

for ii in range(nrow):
     print("\r{0} : {1}".format(ii+1, nrow)),
     sys.stdout.flush()
     for jj in range(ncol):

        xx, yy, zz = colin_inv_rad(params, jj - ncol // 2, nrow // 2 - ii, rs)

        rind  = (yy - mm[0][1]).round().astype(int)
        cind  = (xx - mm[0][0]).round().astype(int)

        tind  = (rind >= 0) & (cind >= 0) & (rind < rast.shape[0]) & \
            (cind < rast.shape[1])
        rind  = rind[tind]
        cind  = cind[tind]
        xx    = xx[tind]
        yy    = yy[tind]
        zz    = zz[tind]
        tall  = rast[rind, cind] > zz

	if tall.size == 0:
            continue
        if tall.max() == False:
            continue

        dd    = (xx[tall] - x0)**2 + (yy[tall] - y0)**2

	close = np.arange(rind.size)[tall][dd.argmin()]

        #index = [rind[close], cind[close]]
       		
	xgrid[ii, jj] = xx[close]
	ygrid[ii, jj] = yy[close]
	dgrid[ii, jj] = dd.min()**0.5

# -- visualize a distance grid
#timg = np.minimum(img[1:], img[:-1])

# -- saving outputs to file
np.save(os.path.join(os.environ["REBOUND_WRITE"], 'xgrid.npy'), xgrid)
np.save(os.path.join(os.environ["REBOUND_WRITE"], 'ygrid.npy'), ygrid)
np.save(os.path.join(os.environ["REBOUND_WRITE"], 'dgrid.npy'), dgrid)
