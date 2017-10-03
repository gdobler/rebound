#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import datetime
import numpy as np
import settings
from utils import read_hour_stack


# -- utilities
data_dir = settings.STACKED_HSI_FILEPATH
opath = settings.HSI_OUT
date_list = []
date_hold = ''

for i in sorted(os.listdir(data_dir))[1:]:
    fdate = i.split('_')[-2]

    if fdate == date_hold:
        data = (read_hour_stack(data_dir, i, settings.STACKED_SHAPE))*1.0/10
        date_list.append(data.astype(np.uint16))
        last = fdate
    
    else:
        if date_hold != '':
            stack_start = time.time()
            print('Time to read {}: {}'.format(last,stack_start-read_start))
            print('Stacking {} ...'.format(last))
            stack = reduce(np.add, date_list)

            stack.transpose(2, 0, 1)[..., ::-1] \
            .flatten().tofile(os.path.join(opath, last+'_stacked.raw'))
            print("Time to stack {}: {}".format(last,time.time()-stack_start))

        date_hold = fdate
        date_list = []

        print('Reading in HSI scans from {}...'.format(fdate))
        read_start = time.time()

        data = (read_hour_stack(data_dir, i, settings.STACKED_SHAPE))*1.0/10
        date_list.append(data.astype(np.uint16))
