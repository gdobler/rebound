#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def label_pixels(img, n, img_type, src, clip=None):
    '''
    Select src vs non-src labels.

    Parameters:
    ----------

    ipath: 2-d numpy array (img file)
        A processed image file (cleaned, stacked, etc)

    img_type: str
        "bb" or "hsi"

    src: bool
        True if labeling sources, otherwise False for non-sources.

    Returns:
        Saves to disk numpy array of labels for <img_type> <src>
    '''

    plt.imshow(img, clim=[clip[0],clip[1]])

    x = plt.ginput(n, show_clicks=True, mouse_pop=3)

    if src:
        np.save(os.path.join(os.environ['REBOUND_WRITE'],'circadian','{}_src_labels.npy'.format(img_type)),x)

    else:
        np.save(os.path.join(os.environ['REBOUND_WRITE'],'circadian','{}_nosrc_labels.npy'.format(img_type)),x)

    plt.show()