#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import numpy as np
from utils import read_hyper

# -- utilities
rr    = [900, 1200]
cr    = [1400, 2200]
dpath = os.path.join(os.environ["REBOUND_DATA"], "hsi", "2017")

# -- get the full file list
flist = sorted(glob.glob(os.path.join(dpath, "*", "*", "*.raw")))

# -- test it
t0   = time.time()
data = np.save("foo.npy", 
               read_hyper(flist[3]).data[:, rr[0]:rr[1], cr[0]:cr[1]])
dt   = t0 - time.time()
print("elapsed time {0}s".format(dt))
