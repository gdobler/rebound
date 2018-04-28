#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import numpy as np
import utils

dpath = os.path.join(os.environ['REBOUND_DATA'], 'hsi', '2017')

nights = [('07', '29'), ('07', '30'), ('08', '01'), ('08', '02'), ('08', '03'), ('08', '04'), ('08', '05'),
          ('08', '06'), ('08', '07'), ('08', '08'), ('08',
                                                     '09'), ('08', '10'), ('08', '11'), ('08', '12'),
          ('08', '13'), ('08', '14'), ('08', '15'), ('08', '16'), ('08', '17'), ('08', '18'), ('08', '19')]

# because we can't assume the camera takes HSI scans at precisely the same times every night, we'll
# base our interval on number of scans rather than timestamp. i.e. first scan of night, 15th scan of
# the night, and 50th scan of the night.


def load_and_stack(bin_size = 8):
    flist = [[os.path.join(dpath, n[0], n[1], i) for i in sorted(os.listdir(os.path.join(
        dpath, n[0], n[1]))) if i.split('.')[-1] == "raw"] for n in nights[:10]]

    flist = sorted([i[j] for j in [0, 14, 49] for i in flist])

    sh = [(i['nwav'], i['nrow'], i['ncol'])
          for i in [utils.read_header(f.replace('raw', 'hdr')) for f in flist]]

    mincol = min([i[2] for i in sh])

    data = np.empty((sh[0][1]*mincol, len(flist) * (sh[0][0]//bin_size)), dtype=np.uint16)

    idx = 0

    for i in range(len(flist))[:1]:

        print('Loading {}...'.format(i))

        # data[idx:idx+sh[0][0], :, :] = np.memmap(flist[i], np.uint16, mode='r').reshape(
        #     sh[i][2], sh[i][0], sh[i][1])[0:mincol, :, ::-1].transpose(1, 2, 0)

        f = np.memmap(flist[i], np.uint16, mode='r').reshape(sh[i][2], sh[i][0], sh[i][1])[0:mincol, :, ::-1].transpose(1, 2, 0).reshape(sh[i][0],sh[i][1]*mincol).T

        data[:,idx:idx+sh[0][0]//bin_size] = f.reshape(f.shape[0],f.shape[1]//bin_size,bin_size).mean(2)


        idx += sh[0][0]//bin_size

    np.save(os.path.join(os.environ['HOME_UO'],
                         'circadian', 'hsi_temp_stack.npy'), data)

    return
