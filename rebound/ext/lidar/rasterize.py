#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils import get_tile_list

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
    print("working on tile {0:3} of {1}".format(ii+1, nfiles))
    tile = np.load(fname)

    # -- snap to grid
    rind = (tile[:, 1] - mm[0][1]).round().astype(int)
    cind = (tile[:, 0] - mm[0][0]).round().astype(int)

    # -- get the counts in each bin
    tcnt = np.bincount(cind + rind * ncol)

    # -- update the raster
    rast[rind, cind] += tile[:, 2]
    cnts[rind, cind] += tcnt[tcnt > 0]

