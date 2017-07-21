#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import utils

def stack_hsi(dpath):
    # -- get the file list
    #dpath = os.path.join(os.environ["REBOUND_DATA"], "slow_hsi_scans")
    flist = sorted(glob.glob(os.path.join(dpath, "*.raw")))

    # -- create the memory maps
    cubes = [utils.read_hyper(f) for f in flist]

    # -- get the minimum number columns
    mincol = min([cube.ncol for cube in cubes])

    # -- initialize the stack
    print("initializing stack...")
    stack = np.zeros((cube.nwav, cube.nrow, mincol), dtype=np.uint16)

    # -- loop through cubes and sum
    for cube in cubes:
        print("adding {0}".format(cube.filename))
        stack[...] += cube.data[..., :mincol]

    return stack/len(cubes)