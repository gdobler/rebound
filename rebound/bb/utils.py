#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import numpy as np
import pandas as pd
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

def clip_labels(cliptype='hsi', light_class='all', lower_thresh=20, upper_sig=2):

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
        new_hsi = bb_settings.FINAL_MASK[1000:1100, 1610:1900].copy()
        hsi_l, hsi_s = np.unique(new_hsi, return_counts=True)

        return hsi_l

    elif cliptype == 'manual':  # manually set lower/upper bounds
        # upper thresh
        thresh = bb_settings.SIZES[1:].mean() + bb_settings.SIZES[1:].std()*upper_sig

        size_msk = (bb_settings.SIZES < thresh) & (bb_settings.SIZES > lower_thresh)

        return bb_settings.LABELS[size_msk]


def load_lc(cube=True, clip=None):
    """
    Loads previously extracted lightcuves. 
    If cube = False, returns a 2-d array time series (num total multi night timesteps x num sources)
    If cube = True, it does so three-dimensionally by night (num nights x timestep/night x num sources)
    """

    # get
    if cube:
        curves = np.empty((bb_settings.NUM_CURVES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))

        nidx = 0

        for i in sorted(os.listdir(bb_settings.CURVES_FILEPATH)):
            curves[nidx, :, :] = (np.load(os.path.join(bb_settings.CURVES_FILEPATH, i)))
            nidx += 1

    else:
        all_curves = []

        for i in sorted(os.listdir(bb_settings.CURVES_FILEPATH)):
            all_curves.append(np.load(os.path.join(bb_settings.CURVES_FILEPATH, i)))

        curves = np.concatenate(all_curves, axis=0)

    if clip is not None:

        if cube:
            return curves[:, :, clip]
        else:
            return curves[:, clip]

    else:
        return curves


def load_onoff(cube=True, clip=None):

    if cube:
        ons = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))
        offs = np.empty((bb_settings.NUM_EDGES, bb_settings.CURVE_LENGTH, len(bb_settings.LABELS[1:])))

        nidx = 0
        fidx = 0

        for i in sorted(os.listdir(bb_settings.EDGE_PATH)):
            if i.split('_')[-1] == 'ons.npy':
                ons[nidx, :, :] = (np.load(os.path.join(bb_settings.EDGE_PATH, i)))
                nidx += 1

            elif i.split('_')[-1] == 'offs.npy':
                offs[fidx, :, :] = (np.load(os.path.join(bb_settings.EDGE_PATH, i)))
                fidx += 1

    else:
        all_ons = []
        all_offs = []

        for i in sorted(os.listdir(bb_settings.EDGE_PATH)):
            if i.split('_')[-1] == 'ons.npy':
                all_ons.append(np.load(os.path.join(bb_settings.EDGE_PATH, i)))

            elif i.split('_')[-1] == 'offs.npy':
                all_offs.append(np.load(os.path.join(bb_settings.EDGE_PATH, i)))

        ons = np.concatenate(all_ons, axis=0)

        offs = np.concatenate(all_offs, axis=0)

    if clip is not None:

        if cube:
            return ons[:, :, clip], offs[:, :, clip]
        else:
            return ons[:, clip], offs[:, clip]

    else:
        return ons, offs


def plot(data=bb_settings.LABELS_MASK, clip=None):
    if clip is not None:
        final_msk = np.isin(data, clip)

        data[~final_msk] = 0

    plotting.quick_source_info(data, clim=None, oname=None)

    return





