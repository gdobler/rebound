#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import settings
import numpy as np


def read_hour_stack(input_dir, rawfile, sh):
    '''
    Reads in raw file already stacked by hour.

    Parameters:
    -----------
    input_dir : str
        Filepath for directory with stacked HSI scans.

    rawfile : str
            Filename for raw file of hourly stacked HSI scan.
            Should be format: "night_stack_ncube10_[yearmonthdate]_[hour].raw"

    sh : tuple (Default reads in (848, 1600, 3194))
            Desired file shape (nwav, nrow, ncol). 


    Returns
    -------
    cube = a numpy memmap of the HSI scan in shape (nwav, nrow, ncol)
    '''

    fpath = os.path.join(input_dir, rawfile)

    return np.memmap(fpath, np.uint16, mode='r').reshape(
    	sh[2], sh[0], sh[1])[:, :, ::-1].transpose(1, 2, 0)
