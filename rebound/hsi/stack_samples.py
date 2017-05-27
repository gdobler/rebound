#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from utils import read_hyper

# -- get the file list
dpath = os.path.join(os.environ["REBOUND_DATA"], "slow_hsi_scans")
flist = sorted(glob.glob(os.path.join(dpath, "*.raw")))

# -- create the memory maps
cubes = [read_hyper(f) for f in flist]

# -- get the minimum number columns
mincol = min([cube.ncol for cube in cubes])

# -- initialize the stack
print("initializing stack...")
stack = np.zeros((cube.nwav, cube.nrow, mincol), dtype=np.uint16)

# -- loop through cubes and sum
for cube in cubes:
    print("adding {0}".format(cube.filename))
    stack[...] += cube.data[..., :mincol]

# -- write to original .raw format
opath  = os.path.join(os.environ["REBOUND_WRITE"], oname)
oname  = "_".join(os.path.split(flist[0]).split("_")[:-1]) + "_stack.raw"
stack.transpose(2, 0, 1)[..., ::-1] \
    .flatten().tofile(os.path.join(opath, oname))
