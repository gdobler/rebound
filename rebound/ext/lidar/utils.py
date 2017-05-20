#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def get_tile_list():
    """
    Get the file name list for the 3D model tiles.
    """
    dpath  = os.path.join(os.environ["REBOUND_DATA"], "data_3d", "data_npy")
    fnames = [os.path.join(dpath, i) for i in sorted(os.listdir(dpath))]

    return fnames

