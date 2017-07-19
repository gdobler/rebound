#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import correlate1d

# global variables
# location of BK bband images
DATA_FILEPATH = os.path.join(os.environ['REBOUND_WRITE'], 'lightcurves') # location of lightcurves
IMG_SHAPE = (3072, 4096)  # dimensions of BK raw images


def edge(curve,output_dir=None):
    """
    Detect the on/off transitions for lightcurves and write to a file.

    Takes as input name of the lightcurve file as located in globally defined DATA_FILEPATH

    Returns Numpy array with 
    """
    start = time.time()
    # -- utilities
    width        = 30
    delta        = 2
    sig_clip_amp = 2.0
    sig_peaks    = 0.0
    sig_xcheck   = 2.0
    month,night  = curve.split('_')[-2],curve.split('_')[-1].split('.')[0]
    
    # -- read in lightcurves
    lcs = np.load(os.path.join(DATA_FILEPATH,curve))

    # -- generate a mask
    print("generating mask for {} {}...".format(month,night))
    msk = gf((lcs > -9999).astype(float), (width, 0)) > 0.9999
    
    # -- convert to smooth the lightcurves (taking into account the mask)
    msk_sm = gf(msk.astype(float), (width, 0))
    lcs_sm = gf(lcs * msk, (width, 0)) / (msk_sm + (msk_sm == 0))
    
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
    
    # -- tag the potential ons and offs
    tags_on  = np.zeros(lcs_gd.shape, dtype=bool)
    tags_off = np.zeros(lcs_gd.shape, dtype=bool)
    
    print("finding extrema...")
    tags_on[1:-1]  = (lcs_gd - final_avg > 
                          sig_peaks * final_sig)[1:-1] & \
                          (lcs_gd[1:-1] > lcs_gd[2:]) & \
                          (lcs_gd[1:-1] > lcs_gd[:-2]) & \
                          ~lcs_gd.mask[1:-1]
    tags_off[1:-1] = (lcs_gd - final_avg < 
                          -sig_peaks * final_sig)[1:-1] & \
                          (lcs_gd[1:-1] < lcs_gd[2:]) & \
                          (lcs_gd[1:-1] < lcs_gd[:-2]) & \
                          ~lcs_gd.mask[1:-1]
    
    time_extrema = time.time()
    print('Time to find max on/off: {}'.format(time_extrema-time_clip))

    # -- cross check left/right means for robustness to noise
    print("setting up filters...")
    mean_diff  = ((np.arange(2 * width) >= width) * 2 - 1) / float(width)
    mean_left  = 1.0 * (np.arange(2 * width) < width) / float(width)
    mean_right = 1.0 * (np.arange(2 * width) >= width) / float(width)
    
    # calculate mean difference across transition
    lcs_md     = np.abs(correlate1d(lcs, mean_diff, axis=0))
    
    # calculate max std across transition
    lcs_sq     = lcs**2
    lcs_std    = np.sqrt(np.maximum(correlate1d(lcs_sq, mean_left, axis=0) - 
                                    correlate1d(lcs, mean_left, axis=0)**2, 
                                    correlate1d(lcs_sq, mean_right, axis=0) - 
                                    correlate1d(lcs, mean_right, axis=0)**2))
    
    time_cor = time.time()
    print('Time to correlate: {}'.format(time_cor-time_extrema))

    # -- identify all potentially robust transitions and prune list
    good_arr  = lcs_md > sig_xcheck * lcs_std

    pad       = np.zeros((lcs.shape), dtype=bool)
    good_ons  = (pad + tags_on) & good_arr
    good_offs = (pad + tags_off) & good_arr
    
    # -- split nights
    # ons = lcs * good_ons
    # offs = lcs * good_offs

    if output_dir != None:
        print("writing ons/offs to files...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # gaussian difference
        # np.save(os.path.join(
        #     output_dir, 'lightcurves_gd_{}_{}.npy'.format(month, night)), lcs_gd)

        # on tags
        np.save(os.path.join(
            output_dir, '{}_{}_ons.npy'.format(month, night)), good_ons)

        # off tags
        np.save(os.path.join(
            output_dir, '{}_{}_offs.npy'.format(month, night)), good_offs)


        end = time.time()
        print "Total runtime: {}".format(end - start)

        return

    else:
        class output():

            def __init__(self):
                self.curves = lcs
                self.gd = lcs_gd
                self.ons = good_ons
                self.offs = good_offs

        end = time.time()
        print "Total runtime: {}".format(end - start)
        return output()




def multi_nights(output_dir, all_nights=False, nights=None):
    """
    Nights input is str or list of strs, formated as: "06_25"  etc
    """

    start_all = time.time()
    if all_nights:
        for lc in os.listdir(DATA_FILEPATH):
            edge(curve=lc,output_dir=output_dir)

    else:
        for n in nights:
            edge(curve='lightcurves_{}.npy'.format(n),output_dir=output_dir)
            
    print "Total runtime for all nights: {}".format(time.time() - start_all)