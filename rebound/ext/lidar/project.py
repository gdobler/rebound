#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils import read_raster
from colin_inv import *

# -- load the raster
fname = os.path.join(os.environ["REBOUND_WRITE"], "rasters", "MN_raster.bin")
rast  = read_raster(fname)


# -- set the camera parameters
params = np.array([1.46161933, -1.26155483e-02, 2.39850465e-02, 
                   9.87891059e+05, 1.91728245e+05, 4.00726823e+02, 
                   1.63556904e+04])
x0     = params[3]
y0     = params[4]


# -- initialize an image
img = np.zeros((1000, 1000), dtype=float)


# -- for each pixel
#    - identify the x,y coordinates for all zs
#    - convert x,y coordinates to indices
#    - find all x,y coordinates for which z is greater than projectws line
#    - of those, find the closest

zs = np.arange(0, 1000., 0.25)
xx, yy = colin_inv(params, 0.0, 0.0, zs)

mm = [[978979.241501, 194479.07369], [1003555.2415, 220149.07369]]

rind = (yy - mm[0][1]).round().astype(int)
cind = (xx - mm[0][0]).round().astype(int)

tind  = (rind > 0) & (cind > 0)
rind  = rind[tind]
cind  = cind[tind]
xx    = xx[tind]
yy    = yy[tind]
tall  = rast[rind, cind] > zs[tind]
dd    = (xx[tall] - x0)**2 + (yy[tall] - y0)**2
close = np.arange(rind.size)[tall][dd.argmin()]
index = [rind[close], cind[close]]



# -- visualize a distance grid
