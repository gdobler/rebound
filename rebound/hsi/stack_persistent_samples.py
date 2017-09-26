#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import datetime
import numpy as np
from utils import read_hyper, read_header

# -- get the file list and times
dpath = os.path.join(os.environ["REBOUND_DATA"], "hsi", "2017", "08", "21")
flist = np.array(sorted(glob.glob(os.path.join(dpath, "night*.raw"))))
ftime = np.array([os.path.getmtime(i) for i in flist])
fhour = np.array([datetime.datetime.fromtimestamp(i).hour for i in ftime])

# -- utilities
hrs = [21, 22, 23, 0, 1, 2, 3, 4]

# -- loop through hours
for hr in hrs:

    # sub select files
    flist_sub = flist[fhour == hr]
    ftime_sub = ftime[fhour == hr]

    # create the memory maps
    hdrs  = [read_header(f.replace("raw", "hdr")) for f in flist_sub]
    ncube = len(hdrs)

    # get the minimum number columns
    mincol = min([hdr["ncol"] for hdr in hdrs])
    sh     = (hdrs[0]["nwav"], hdrs[0]["nrow"], mincol)

    # read the cubes
    cubes = [read_hyper(f).data[:, :, :mincol] for f in flist_sub]

    # initialize the stack
    print("creating stack...")
    t0    = time.time()
    stack = reduce(np.add, cubes)
    print("  created in {0}s".format(time.time() - t0))

    # extract time parameters from the first file
    tdt  = datetime.datetime.fromtimestamp(ftime_sub[0])
    dstr = "{0:04}{1:02}{2:02}".format(tdt.year, tdt.month, tdt.day)
    tstr = "{0:02}".format(tdt.hour)

    # -- write to original .raw format
    opath = os.path.join(os.environ["REBOUND_DATA"], "hsi")
    oname = "night_stack_ncube{0:02}_{1}_{2}.raw".format(ncube, dstr, tstr)
    stack.transpose(2, 0, 1)[..., ::-1] \
        .flatten().tofile(os.path.join(opath, oname))
