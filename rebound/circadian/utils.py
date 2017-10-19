#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import settings
import numpy as np


def read_hsi(input_dir, rawfile, sh):
    '''
    Reads in raw file with no header.

    Parameters:
    -----------
    input_dir : str
        Filepath for directory with HSI scans.

    rawfile : str
            Filename for raw file of HSI scan.

    sh : tuple (Default reads in (848, 1600, 3194))
            Desired file shape (nwav, nrow, ncol).


    Returns
    -------
    cube = a numpy memmap of the HSI scan in shape (nwav, nrow, ncol)
    '''

    fpath = os.path.join(input_dir, rawfile)

    return np.memmap(fpath, np.uint16, mode='r').reshape(
    	sh[2], sh[0], sh[1])[:,:,::-1].transpose(1, 2, 0)


def mask_box(input_mask):
    '''
    Create a bounding box around nonzero values in a numpy mask (e.g. Gowanus).

    Parameters
    ----------
    input_mask = 2-d numpy array
        The numpy mask you want to create a bounding box within.

    Returns
    -------
    Row min, row max, column min, column max for bounding box area with 
    nonzero values.
    '''
    rows = np.any(input_mask, axis = 1)
    cols = np.any(input_mask, axis = 0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def sigma_clipping(input_file, sig=3, iter=10):
    '''
    Takes numpy memmap of an HSI scan and adjusts median by clipping extremes 
    and iterating in order to clear HSi scan.

    Parameters
    ----------

    input_file : numpy array
        Datacube of the HSI scan to be cleaned, shape (nwav, nrows, ncols)

    sig : int (default 3)
        Number of standard deviations to clip in each pass.

    iter : int (default 10)
        Number of passess before final subtraction.
    '''
    
    # -- compute the gaussian difference (using a masked array now)
    lcs_gd = np.ma.zeros(lcs_sm.shape, dtype=lcs_sm.dtype)
    lcs_gd[delta // 2: -delta // 2] = lcs_sm[delta:] - lcs_sm[:-delta]
    
    # -- set the gaussian difference mask
    lcs_gd.mask = np.zeros_like(msk)
    lcs_gd.mask[delta // 2: -delta // 2] = ~(msk[delta:] * msk[:-delta])
    
    time_clip = time.time()

    # -- sigma clip and reset the means, standard deviations, and masks
    print("sigma clipping ...")
    tmsk = lcs_gd.mask.copy()
    for _ in range(10):
        avg             = lcs_gd.mean(0)
        sig             = lcs_gd.std(0)
        lcs_gd.mask = np.abs(lcs_gd - avg) > sig_clip_amp * sig
    final_avg = lcs_gd.mean(0).data
    final_sig = lcs_gd.std(0).data
    lcs_gd.mask = tmsk    