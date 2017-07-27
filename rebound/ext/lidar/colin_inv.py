#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def colin_inv(params, xim, yim, z):
    """
    This function returns inverse of the colinearity equations.  Namely 
    given camera paramters and a location on the image, return the x, y
    coordinate for a given height z.
    NOTE: this function is vectorized in z.

    Parameters
    ----------
    params : tuple
        The camera parameters kappa, phi, omega, xs, ys, zs, and f.
    xim : float
        The x location in an image.
    yim : float
        The y location in an image.
    z : float
        The height of the projected point.

    Returns
    -------
    x, y : tuple
        The x and y locations of the projected points.  These are either 
        floats, or ndarrays
    """

    # -- Unwrap params
    kappa, phi, omega, xs, ys, zs, f = params

    omega = float(omega)
    phi   = float(phi) + 0.5 * np.pi
    kappa = float(kappa)
    xs    = float(xs)
    ys    = float(ys)
    zs    = float(zs)
    f     = float(f)

    # -- utils
    co = np.cos(omega)
    so = np.sin(omega)
    cp = np.cos(phi)
    sp = np.sin(phi)
    ck = np.cos(kappa)
    sk = np.sin(kappa)

    a1 =  cp * ck + sp * so * sk
    b1 =  cp * sk + sp * so * ck
    c1 =  sp * co
    a2 = -co * sk
    b2 =  co * ck
    c2 =  so
    a3 =  sp * ck + cp * so * sk
    b3 =  sp * sk - cp * so * ck
    c3 =  cp * co

    d1 = a3 * xim + f * a2
    d2 = b3 * xim + f * b2
    d3 = c3 * xim + f * c2
    d4 = a3 * yim - f * a1
    d5 = b3 * yim - f * b1
    d6 = c3 * yim - f * c1

    # -- return the x, y location of the projected point
    x = (d6 - d3 * d5 / d2) / (d1 * d5 / d2 - d4) * (z - zs) + xs
    y = (d6 - d3 * d4 / d1) / (d2 * d4 / d1 - d5) * (z - zs) + ys

    return x, y



def colin_inv_rad(params, xim, yim, r):
    """
    This function returns inverse of the colinearity equations.  Namely 
    given camera paramters and a location on the image, return the x, y, z
    coordinate for a given distance from the camera r.
    NOTE: this function is vectorized in z.

    Parameters
    ----------
    params : tuple
        The camera parameters kappa, phi, omega, xs, ys, zs, and f.
    xim : float
        The x location in an image.
    yim : float
        The y location in an image.
    r : float
        The distance from the camera.

    Returns
    -------
    x, y, z : tuple
        The x and y locations of the projected points.  These are either 
        floats, or ndarrays
    """

    # -- careful about zeros
    xim += 1e-3
    yim += 1e-3

    # -- Unwrap params
    kappa, phi, omega, xs, ys, zs, f = params

    omega = float(omega)
    phi   = float(phi) + 0.5 * np.pi
    kappa = float(kappa)
    xs    = float(xs)
    ys    = float(ys)
    zs    = float(zs)
    f     = float(f)

    # -- utils
    co = np.cos(omega)
    so = np.sin(omega)
    cp = np.cos(phi)
    sp = np.sin(phi)
    ck = np.cos(kappa)
    sk = np.sin(kappa)

    a1 =  cp * ck + sp * so * sk
    b1 =  cp * sk + sp * so * ck
    c1 =  sp * co
    a2 = -co * sk
    b2 =  co * ck
    c2 =  so
    a3 =  sp * ck + cp * so * sk
    b3 =  sp * sk - cp * so * ck
    c3 =  cp * co

    d1 = a3 * xim + f * a2
    d2 = b3 * xim + f * b2
    d3 = c3 * xim + f * c2
    d4 = a3 * yim - f * a1
    d5 = b3 * yim - f * b1
    d6 = c3 * yim - f * c1

    e1 = (d6 * d2 - d3 * d5) / (d1 * d5 - d4 * d2)
    e2 = (d6 * d1 - d3 * d4) / (d2 * d4 - d5 * d1)
    e3 = 1.0 / np.sqrt(1 + e1 * e1 + e2 * e2)

    # -- return the x, y, z location of the projected point
    x = e1 * e3 * r + xs
    y = e2 * e3 * r + ys
    z = e3 * r + zs
    #xyz = np.vstack([x, y, z]).T

    #return xyz
    return x, y, z
