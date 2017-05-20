#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

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


def read_raw(fname, sh=[3072, 4096], rgb=False):
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

    Returns
    -------
    ndarray
        The output image; either 2D or 3D if rgb is set to True.
    """

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
        The upper and lower limit to clip the image

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
