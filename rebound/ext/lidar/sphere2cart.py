#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def sphere2cart(rad, theta, phi, rad0=0.0, theta0=0.0, phi0=0.0):
    """
    Convert spherical coordinates to cartesian coordinates.
    NOTE: the range of theta is [0, 2pi] and the range of phi is [-pi, pi].

    Parameters
    ----------
    rad : float or ndarray
        Radius in arbitrary units
    theta : float or ndarray
        Azimuthal angle in radians between 0 and 2pi.
    phi : float or ndarray
        Polar angle in radians between -pi and pi.
    rad0 : float, optional 
        Origin offset radius.
    theta0 : float, optional
        Origin azimuthal angle.
    phi0 : float, optional
        Origin polar angle.

    Returns
    -------
    xyz : tuple
        The cartesian coordinates.
    """

    # -- utilities
    rnorm = rad - rad0
    tnorm = theta - theta0
    pnorm = -(phi - phi0) + 0.5 * np.pi
    rsinp = rnorm * np.sin(pnorm)

    # -- return cartesian
    return rsinp * np.cos(tnorm), rsinp * np.sin(tnorm), rnorm * np.cos(pnorm)
