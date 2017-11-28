#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import settings
import numpy as np
from scipy import ndimage as nd


def read_header(hdrfile, verbose=False):
    """
    Read a Middleton header file.

    Parameters
    ----------
    hdrfile : str
        Name of header file.
    verbose : bool, optional
        If True, alert the user.

    Returns
    -------
    dict : dict
        A dictionary continaing the number of rows, columns, and wavelengths
        as well as an array of band centers.
    """

    # -- alert
    if verbose:
        print("reading and parsing {0}...".format(hdrfile))

    # -- open the file and read in the records
    recs = [rec for rec in open(hdrfile)]

    # -- parse for samples, lines, bands, and the start of the wavelengths
    for irec, rec in enumerate(recs):
        if 'samples' in rec:
            samples = int(rec.split("=")[1])
        elif 'lines' in rec:
            lines = int(rec.split("=")[1])
        elif 'bands' in rec:
            bands = int(rec.split("=")[1])
        elif "Wavelength" in rec:
            w0ind = irec+1

    # -- parse for the wavelengths
    waves = np.array([float(rec.split(",")[0]) for rec in 
                      recs[w0ind:w0ind+bands]])

    # -- return a dictionary
    return {"nrow":samples, "ncol":lines, "nwav":bands, "waves":waves}


def read_raw(rawfile, shape, hyper=False, verbose=False):
    """
    Read a Middleton raw file.

    Parameters
    ----------
    rawfile : str
        The name of the raw file.
    shape : tuple
        The output shape of the data cube (nwav, nrow, ncol).
    hyper : bool, optional
        Set this flag to read a hyperspectral image.
    verbose : bool, optional
        Alert the user.

    Returns
    -------
    memmap : memmap
        A numpy memmap of the datacube.
    """

    # -- alert
    if verbose:
        print("reading {0}...".format(rawfile))

    # -- read either broadband or hyperspectral image
    if hyper:
        return np.memmap(rawfile, np.uint16, mode="r") \
                .reshape(shape[2], shape[0], shape[1])[:, :, ::-1] \
                .transpose(1, 2, 0)
    else:
        return np.memmap(rawfile, np.uint8, mode="r") \
            .reshape(shape[1], shape[2], shape[0])[:, :, ::-1]


def read_hyper(fpath, fname=None, full=True):
    """
    Read a full hyperspectral scan (raw and header file).

    Parameters
    ----------
    fpath : str
        Either the full name+path of the raw file or the path of the raw file.
        If the latter, fname must be supplied.
    fname : str, optional
        The name of the raw file (required if fpath is set to a path).

    full : bool, optional
        If True, output a class containing data and supplementary information.
        If False, output only the data.

    Returns
    -------
    output or memmap : class or memmap
        If full is True, a class containing data plus supplementary 
        information.  If full is False, a memmap array of the data.
    """

    # -- set up the file names
    if fname is not None:
        fpath = os.path.join(fpath, fname)

    # -- read the header
    hdr = read_header(fpath.replace("raw", "hdr"))
    sh  = (hdr["nwav"], hdr["nrow"], hdr["ncol"])

    # -- if desired, only output data cube
    if not full:
        return read_raw(fpath, sh, hyper=True)

    # -- output full structure
    class output():
        def __init__(self, fpath):
            self.filename = fpath
            self.data     = read_raw(fpath, sh, hyper=True)
            self.waves    = hdr["waves"]
            self.nwav     = sh[0]
            self.nrow     = sh[1]
            self.ncol     = sh[2]

    return output(fpath)


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


def augment_mask(labels, min_thresh=3):
    '''
    Filter out broadband mask sources below minimum threshold and add pixels to left, right, up, down.
    '''
    uniq, size = np.unique(labels, return_counts=True)
    in_labels = lambda x: x in uniq[size > min_thresh][1:]

    labels *= np.array([in_labels(x) for x in labels.flatten()]).reshape(labels.shape)

    # need to vectorize!!
    m = labels.copy()

    for i in range(1, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            if labels[i, j] == 0:
                m[i, j] = labels[i-1, j]

    n = m.copy()

    for i in range(0, labels.shape[0]):
        for j in range(1, labels.shape[1]):
            if m[i, j] == 0:
                n[i, j] = m[i, j-1]

    n = n[::-1,::-1]
    o = n.copy()

    for i in range(1, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            if n[i, j] == 0:
                o[i, j] = n[i-1, j]

    p = o.copy()

    for i in range(0, labels.shape[0]):
        for j in range(1, labels.shape[1]):
            if o[i, j] == 0:
                p[i, j] = o[i, j-1]

    return o[::-1,::-1]




def sigma_clip(input_file, ax, sig_amp=3, iter=10, median_clip=False):
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
    
    # -- create mask
    data = np.ma.zeros(input_file.shape, dtype=input_file.dtype)
    data[:,:,:] = input_file.copy()

    data.mask = np.zeros_like(input_file)

    temp = data.mask.copy()

    # -- sigma clip along rows and reset the means, standard deviations, and masks
    for _ in range(iter):
        if median_clip:
            avg         = np.median(data, axis=ax, keepdims=True)
        else:
            avg             = np.mean(data, axis=ax, keepdims=True)
        sig             = np.std(data, axis=ax, keepdims=True)
        data.mask = np.abs(data - avg) > sig_amp * sig


    return input_file - np.median(data, axis=ax, keepdims=True).data

def mean_spectra(scans, labels, gow=False):
    '''
    Takes a file of stacked HSI scans and an array of labels (i.e. broadband mask)
    and finds the mean spectra for the sources in lables.

    Scan and labels must share nrows and ncols.

    Parameters:
    -----------
    scans : raw file
        Stacked HSI scans data cube (nwav x nrows x ncols). Assumes scans have been suffificently cleaned.

    labels : numpy array
        2-d array of pixels of labeled sources (nrows x ncols)

    gow : bool (default False)
        If True, slices labels to dimensions of Gowanus. Scans must also be same dimension.

    Returns:
    --------
    Dictionary with keys as source labels and values are 1-d arrays of spectra for the source
    (light intensity over 848 wavelength channels).
    '''

    # utilities
    gow_row = (900, 1200)
    gow_col = (1400, 2200)

    if gow:
        labels = labels[gow_row[0]:gow_row[1], gow_col[0]:gow_col[1]]

    idx = np.unique(labels)[1:] # array of labels without 0

    src_spectra = []

    for i in range(scans.shape[0]):
        scan_mu = nd.measurements.mean(scans[i, :, :], labels, idx)
        scan_mu = scan_mu.reshape(scan_mu.shape[0], 1)
        src_spectra.append(scan_mu)

    src_array = np.concatenate(src_spectra, axis = 1)

    s_dict = {}
    for s in range(idx.shape[0]):
        s_dict[idx[s]] = src_array[s, :]

    return s_dict



