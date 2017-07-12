#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import correlate1d

def detect_onoff(lc_file_in, sig_peaks =0.0, multi=False):
    """
    Detect the on/off transitions for lightcurves and write to a file.
    """

    # -- utilities
    width        = 30
    delta        = 2
    sig_clip_amp = 2.0
    sig_peaks    = sig_peaks
    sig_xcheck   = 2.0
    
    # -- read in lightcurves
    lcs = np.load(lc_file_in)
    
    # -- generate a mask
    print("generating mask...")
    msk = gf((lcs > -9999).astype(float), (width, 0)) > 0.9999
    
    # -- convert to smooth the lightcurves (taking into account the mask)
    print("smoothing lightcurves...")
    msk_sm = gf(msk.astype(float), (width, 0))
    lcs_sm = gf(lcs * msk, (width, 0)) / (msk_sm + (msk_sm == 0))
    
    # -- compute the gaussian difference (using a masked array now)
    print("computing gaussian differences...")
    lcs_gd = np.ma.zeros(lcs_sm.shape, dtype=lcs_sm.dtype)
    lcs_gd[delta // 2: -delta // 2] = lcs_sm[delta:] - lcs_sm[:-delta]
    
    # -- set the gaussian difference mask
    print("resetting the mask...")
    lcs_gd.mask = np.zeros_like(msk)
    lcs_gd.mask[delta // 2: -delta // 2] = ~(msk[delta:] * msk[:-delta])
    
    # -- get the indices of the date separators and create the individual dates
    if multi:
        print("splitting dates...")
        dind_lo = list(split_days(file_index))
        dind_hi = dind_lo[1:] + [lcs_gd.shape[0]]
        nights  = [lcs_gd[i:j] for i, j in zip(dind_lo, dind_hi)]
        nnights = len(nights)
    
        # -- sigma clip and reset the means, standard deviations, and masks
        avgs = []
        sigs = []
        for ii in range(nnights):
            print("sigma clipping night {0} of {1}...".format(ii + 1, nnights))
            tmsk = nights[ii].mask.copy()
            for _ in range(10):
                avg             = nights[ii].mean(0)
                sig             = nights[ii].std(0)
                nights[ii].mask = np.abs(nights[ii] - avg) > sig_clip_amp * sig
            avgs.append(nights[ii].mean(0).data)
            sigs.append(nights[ii].std(0).data)
            nights[ii].mask = tmsk
        
        # -- tag the potential ons and offs
        tags_on  = [np.zeros(i.shape, dtype=bool) for i in nights]
        tags_off = [np.zeros(i.shape, dtype=bool) for i in nights]
        
        for ii in range(nnights):
            print("finding extrema for night {0} of {1}".format(ii + 1, nnights))
            tags_on[ii][1:-1]  = (nights[ii] - avgs[ii] > 
                                  sig_peaks * sigs[ii])[1:-1] & \
                                  (nights[ii][1:-1] > nights[ii][2:]) & \
                                  (nights[ii][1:-1] > nights[ii][:-2]) & \
                                  ~nights[ii].mask[1:-1]
            tags_off[ii][1:-1] = (nights[ii] - avgs[ii] < 
                                  -sig_peaks * sigs[ii])[1:-1] & \
                                  (nights[ii][1:-1] < nights[ii][2:]) & \
                                  (nights[ii][1:-1] < nights[ii][:-2]) & \
                                  ~nights[ii].mask[1:-1]
    else:
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
        
        print("finding extrema for night...")
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
    
    return lcs, lcs_gd, tags_on,tags_off

    # -- cross check left/right means for robustness to noise
    print("setting up filters...")
    mean_diff  = ((np.arange(2 * width) >= width) * 2 - 1) / float(width)
    mean_left  = 1.0 * (np.arange(2 * width) < width) / float(width)
    mean_right = 1.0 * (np.arange(2 * width) >= width) / float(width)
    print("calculating mean difference across transition...")
    lcs_md     = np.abs(correlate1d(lcs, mean_diff, axis=0))
    print("calculating max standard deviation across transition...")
    lcs_sq     = lcs**2
    lcs_std    = np.sqrt(np.maximum(correlate1d(lcs_sq, mean_left, axis=0) - 
                                    correlate1d(lcs, mean_left, axis=0)**2, 
                                    correlate1d(lcs_sq, mean_right, axis=0) - 
                                    correlate1d(lcs, mean_right, axis=0)**2))
    
    # -- identify all potentially robust transitions and prune list
    #    NB: careful about handling partial initial days.
    print("pruning ons and offs for statistical significance...")
    pad       = np.zeros((dind_lo[0], lcs.shape[1]), dtype=bool)
    good_arr  = lcs_md > sig_xcheck * lcs_std
    good_ons  = np.vstack([pad] + tags_on) & good_arr
    good_offs = np.vstack([pad] + tags_off) & good_arr
    
    # -- split nights
    ons  = [good_ons[i:j] for i, j in zip(dind_lo, dind_hi)]
    offs = [good_offs[i:j] for i, j in zip(dind_lo, dind_hi)]
    lcsn = [lcs[i:j] for i, j in zip(dind_lo, dind_hi)]
    
    # -- write on/offs to file
    print("writing ons/offs to files...")
    np.save("output/good_ons_{0:04}.npy".format(file_index), good_ons)
    np.save("output/good_offs_{0:04}.npy".format(file_index), good_offs)

    return


def detect_all_onoff():
    """
    Run all detections
    """

    for find in range(20):
        print("running file index {0:2}".format(find))
        detect_onoff(find)

    return