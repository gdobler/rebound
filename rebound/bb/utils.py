#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import numpy as np
import pandas as pd
import cPickle as pickle
import plotting
import bb_settings
import datetime
from dateutil import tz
import scipy.misc as spm

def rg8_to_rgb(img):
    """
    Convert RG8 img to RGB image.  NOTE: no interpolation is performed, 
    only array slicing and depth stacking.  If the shape of the input 
    image is [Nr, Nc], the shape of the output image is [Nr/2, Nc/2, 3].

    Parameters:
    -----------
    img : ndarray
        An RG8 monochrome image.

    Returns:
    -------
    ndarray
        The RGB image.
    """

    red = img[::2, 1::2]
    grn = img[::2, ::2] #// 2 + img[1::2, 1::2] // 2
    blu = img[1::2, ::2]

    return np.dstack((red, grn, blu))


def read_raw(fname, sh=[3072, 4096], rgb=False, usb=False):
    """
    Read in a raw image file.  Assumes monochrome RG8.

    Parameters
    ----------
    fname : str
        The name (full path) of the image to read.
    sh : list
        The number of rows and columns of the output image.
    rgb : bool, optional
        Flag to convert to RGB.
    usb : bool, optional
        Read in an image file from the old USB camera.

    Returns
    -------
    ndarray
        The output image; either 2D or 3D if rgb is set to True.
    """

    if usb:
        return np.fromfile(fname, dtype=np.uint8) \
            .reshape(2160, 4096, 3)[..., ::-1]

    if rgb:
        return rg8_to_rgb(np.fromfile(fname, dtype=np.uint8).reshape(sh))

    return np.fromfile(fname, dtype=np.uint8).reshape(sh)


def gamma_scale(rgb, gam=0.5, scl=1.0, gray=False):
    """
    Scale and "gamma correct" an rgb image.  Values are clipped at 2^8.

    Parameters
    ----------
    rgb : ndarray
        An 8-bit RGB image.
    gam : float, optional
        The power to which to take the image (applied first).
    scl : float, optional
        The scaling factor for the image (applied second).
    gray : bool, optional
        Apply gray world correction before gamma correction.

    Returns
    -------
    ndarray
        A scaled, gamma corrected RGB image.
    """

    if gray:
        print("GRAY WORLD CORRECTION NOT IMPLEMENTED YET!!!")
        return 0

    return ((rgb / 255.)**gam * 255 * scl).clip(0, 255).astype(np.uint8)


def stretch_image(img, lim):
    """
    Stretch an 8-bit image to limits and return an 8-bit image.
    NOTE: if the limits are outside the range [0, 255], the clipping has
    no effect.

    Parameters
    ----------
    img : ndarray
        An 8-bit image.
    lim : list
        The upper and lower limit to clip the image.

    Returns
    -------
    ndarray
        A stretched 8-bit image.
    """

    # -- clip the image
    cimg = img.clip(lim[0], lim[1]).astype(float)

    # -- rescale to 0 to 255
    cimg -= cimg.min()
    cimg *= 255.0 / cimg.max()

    return cimg.astype(np.uint8)


def convert_raws(path, fac=1, gray=False):
    """
    Convert all raw files in a directory to jpg.  NOTE: Image size 
    and RGB are HARD CODED!

    Parameters
    ----------
    path : str
        Path to raw files.
    fac : int, optional
        Sampling of the image.
    """

    # -- set the scaling
    if gray:
        scl = np.array([0.57857543, 0.96972706, 1.0])
    else:
        scl = np.array([1.0, 1.0, 1.0])

    # -- get the file names
    fnames = [os.path.join(path, i) for i in sorted(os.listdir(path))[100:2700] if 
              ".raw" in i]
    nfiles = len(fnames)

    # -- initialize the image
    sh  = (3072 // (2 * fac), 4096 // (2 * fac), 3)
    img = np.zeros(sh, dtype=np.uint8)

    # -- loop through files and convert
    for ii, fname in enumerate(fnames):
        imname = fname.replace("raw", "jpg")
        if os.path.isfile(imname):
            continue
        if (ii + 1) % 25 == 0:
            print("\rworking on file {0:5} of {1:5}..." \
                      .format(ii + 1, nfiles)), 
            sys.stdout.flush()
        img[...] = read_raw(fname, rgb=True)[::fac, ::fac]
        spm.imsave(imname , stretch_image(gamma_scale((img / scl) \
                       .clip(0, 255).astype(np.uint8), 0.5, 2.0), (60, 200)))
    print("")


def get_timestamp(nights):
    """
    Nights = list of 2-tuples of (month,night): str fomrmat i.e. [("06","25")]
    """
    night_dict = {}

    for n in nights:
        data_dir = os.path.join(bb_settings.DATA_FILEPATH, n[0], n[1])
        night_dict[n] = [datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(
            data_dir, file))) for file in sorted(os.listdir(data_dir))[100:2700]]

    return night_dict


def convert_tstamp(tstamps):
    '''
    Reads in array of naive Unix timestamps and returns array of datetime objects localized to New York time.
    '''
    # function to set local time
    set_tz = np.vectorize(lambda x: datetime.datetime.utcfromtimestamp(x).replace(tzinfo=tz.gettz('America/New York')))

    return set_tz(tstamps)

def clip_labels(cliptype='hsi', light_class='all'):

    # utilities
    lower_thresh = 20
    sig_amp = 2


    if cliptype == 'hsi_mask':  # all sources in HSI mask
        hsi_l, hsi_s = np.unique(bb_settings.FINAL_MASK, return_counts=True)

        return hsi_l

    elif cliptype == 'hsi':  # sources with classified spectra
        if light_class == 'all':
            return bb_settings.SPECTRA_CLASS[bb_settings.SPECTRA_CLASS.columns[-4]].values
        else:
            selector_msk = bb_settings.SPECTRA_CLASS[
                bb_settings.SPECTRA_CLASS.columns[-2]] == light_class

            return bb_settings.SPECTRA_CLASS[selector_msk][bb_settings.SPECTRA_CLASS.columns[-4]].values

    elif cliptype == 'gowanus':  # sources estimated to be in Gowanus public housing
        new_hsi = bb_settings.FINAL_MASK[900:1200, 1400:2200].copy()
        hsi_l, hsi_s = np.unique(new_hsi, return_counts=True)

        return hsi_l

    elif cliptype == 'manual':  # manually set lower/upper bounds
        # upper thresh
        thresh = bb_settings.SIZES[1:].mean() + bb_settings.SIZES[1:].std()*sig_amp

        size_msk = (bb_settings.SIZES < thresh) & (bb_settings.SIZES > lower_thresh)

        return bb_settings.LABELS[size_msk]


def load_lc(cube=True, clip=None, light_class='all'):
    """
    Loads previously extracted lightcuves. 
    If cube = False, returns a 2-d array time series (num total multi night timesteps x num sources)
    If cube = True, it does so three-dimensionally by night (num nights x timestep/night x num sources)

    If clipping, set "clip" to cliptype and if cliptype='hsi', set 'light_class' (default 'all')
    """

    # get
    if cube:
        curves = np.empty((bb_settings.NUM_CURVES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        tstamps = np.empty((bb_settings.NUM_CURVES, bb_settings.CURVE_LENGTH))

        nidx = 0

        for i in sorted(os.listdir(bb_settings.CURVES_FILEPATH)):
            with open(os.path.join(bb_settings.CURVES_FILEPATH, i), 'rb') as file:
                lc,ts = pickle.load(file)

            curves[nidx, :, :] = lc

            tstamps[nidx, :] = ts

            nidx += 1

    else:
        all_curves = []
        all_ts = []

        for i in sorted(os.listdir(bb_settings.CURVES_FILEPATH)):
            with open(os.path.join(bb_settings.CURVES_FILEPATH, i), 'rb') as file:
                lc,ts = pickle.load(file)

            all_curves.append(lc)
            all_ts.append(ts)

        curves = np.concatenate(all_curves, axis=0)
        tstamps = np.concatenate(all_ts)

    if clip is not None:
        clip_idx  = clip_labels(cliptype=clip, light_class=light_class)

        if cube:
            return curves[:, :, clip_idx], tstamps
        else:
            return curves[:, clip_idx], tstamps

    else:
        return curves, tstamps


def load_edges(cube=True, clip=None, light_class='all'):

    num_files = bb_settings.NUM_EDGES

    if cube:
        lcs = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        lcgds = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        ons = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        offs = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        tstamps = np.empty((bb_settings.NUM_CURVES, bb_settings.CURVE_LENGTH))

        nidx = 0

        for i in sorted(os.listdir(bb_settings.EDGE_PATH)):
            with open(os.path.join(bb_settings.EDGE_PATH, i), 'rb') as file:
                lc, lcgd, on, off, ts = pickle.load(file)

            lcs[nidx, :, :] = lc
            lcgds[nidx, :, :] = lcgd
            ons[nidx, :, :] = on
            offs[nidx, :, :] = off
            tstamps[nidx, :] = ts

            nidx += 1

            ratio_done = int(nidx*100.0 / num_files)
            if nidx % 10 == 0:
                print "{} % of nights loaded...".format(ratio_done)

    else:
        lcs = []
        lcgds = []
        ons = []
        offs = []
        tstamps = []

        nidx = 0

        for i in sorted(os.listdir(bb_settings.EDGE_PATH)):
            with opn(os.path.join(bb_settings.EDGE_PATH, i), 'rb') as file:
                lc, lcgd, on, off, ts = pickle.load(file)

            lcs.append(lc)
            lcgds.append(lcgd)
            ons.append(on)
            offs.append(off)
            tstamps.app(ts)

            nidx += 1

            ratio_done = int(nidx*100.0 / num_files)
            if nidx % 10 == 0:
                print "{} % of nights loaded...".format(ratio_done)

        lcs = np.concatenate(lcs, axis=0)
        lcgds = np.concatenate(lcgds, axis=0)
        ons = np.concatenate(ons, axis=0)
        offs = np.concatenate(offs, axis=0)
        tstamps = np.concatenate(tstamps)

    if clip is not None:
        clip_idx  = clip_labels(cliptype=clip, light_class=light_class)

    else:
        clip_idx = np.arange(len(bb_settings.LABELS[1:]))

    if cube:
        return lcs[:,:, clip_idx], lcgds[:,:, clip_idx], ons[:, :, clip_idx], offs[:, :, clip_idx], tstamps
    else:
        return lcs[:, clip_idx], lcgds[:, clip_idx], ons[:, clip_idx], offs[:, clip_idx], tstamps


def last_off(offs, tstamps, clip=None, light_class='all', time_convert=False):
    '''
    Takes edge objects and determines "last off" for each source for each night.

    Parameters:
    -----------
    offs : 3-d np.array
        Array of off indices per night per source (nnights x nobs/night x nsources)

    clip
        If not None, clips sources according to clip_labels() method above.

    Returns:
    --------
        Array of time last off per source per night (nnights x nsources)
    '''
    
    num_files = bb_settings.NUM_EDGES

    last_offs = np.empty((num_files, offs.shape[2]))

    for i in range(num_files):
        if_on = offs[i].T * tstamps[i]

        if_on = np.max(if_on, axis=1)

        if_on[if_on == 0.0] = None

        last_offs[i, :] = if_on

    if time_convert:
        temp = np.ma.zeros(last_offs.shape)
        temp[:,:] = last_offs
        temp.mask = np.zeros_like(last_offs)
        temp.mask[:,:] = np.isnan(last_offs)

        def convert(data):
            try:
                return datetime.datetime.fromtimestamp(data)

            except ValueError:
                return np.ma.masked

        cvect = np.vectorize(convert)

        return cvect(temp).data

    else:
        return last_offs


def plot(data=bb_settings.LABELS_MASK, clip=None):
    if clip is not None:
        final_msk = np.isin(data, clip)

        data[~final_msk] = 0

    plotting.quick_source_info(data, clim=None, oname=None)

    return





