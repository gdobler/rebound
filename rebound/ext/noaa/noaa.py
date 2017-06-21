#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf


class HyperNoaa(object):
    """
    Container for NOAA template spectra measured in the lab.

    E.g., see  http://ngdc.noaa.gov/eog/data/web_data/nightsat/.

    Attributes
    ----------
    fpath : str
        The path to the file.

    flist : str
        The list of NOAA xls file names.

    wavelength : ndarray
        The observed wavelengths in nm.

    data : dict
        A dictionary of dictionaries holding the NOAA intensities. Each key 
        represents a NOAA lighting type and for each of those, each key is an 
        example of that lighting type.
    """

    def __init__(self, fpath=None):

        # -- defaults
        fpath = fpath if fpath else os.path.join(os.getenv("REBOUND_DATA"), 
                                                 "noaa")

        # -- set the data path and file list, and initialize the container
        self.fpath = fpath
        self.flist = ['Fluorescent_Lamps_20100311.xls',
                      'High_Pressure_Sodium_Lamps_20100311.xls',
                      'Incandescent_Lamps_20100311.xls',
                      'LED_Lamps_20100311.xls',
                      'Low_Pressure_Sodium_Lamp_20100311.xls',
                      'Mercury_Vapor_Lamp_20100311.xls',
                      'Metal_Halide_Lamps_20100311.xls',
                      'Oil_Lanterns_20100311.xls',
                      'Pressurized_Gas_Lanterns_20100311.xls',
                      'Quart_Halogen_Lamps_20100311.xls']
        self.data  = {}

        # -- read in the NOAA xls files and convert to ndarrays
        print("NOAA: reading NOAA templates from {0}".format(fpath))

        try:
            noaa = [pd.read_excel(os.path.join(self.fpath,i)) for i in 
                    self.flist]
        except:
            print("NOAA: file read failed!!!")
            return

        for tfile,tdata in zip(self.flist,noaa):
            try:
                self.wavelength
            except:
                self.wavelength = tdata['Wavelength (nm)'].as_matrix()
                self._nwav      = len(self.wavelength)

            tname = '_'.join(tfile.split('_')[:-2]).replace("Quart","Quartz")
            self.data[tname] = {}
            for key in tdata.keys():
                if 'Wavelength' in key:
                    continue
                self.data[tname][key] = tdata[key].as_matrix()[:self._nwav]
                self.data[tname][key][np.isnan(self.data[tname][key])] = 0.0

        # -- some useful data characteristics
        self.row_names = np.array([[i,j] for i in self.data.keys() for j in 
                                   self.data[i].keys()])
        self.rows      = np.zeros([len(self.row_names),self._nwav])

        for ii,jj in enumerate(self.row_names):
            self.rows[ii] = self.data[jj[0]][jj[1]]

        self._min    = 0.0
        self._max    = 2.5
        self._minmax = [self._min,self._max]

        return


    def interpolate(self,iwavelength=None,ltype=None,example=None):
        """
        Linearly interpolate the NOAA spectra onto input wavelngths.

        If the lighting type and example are set, this function will output 
        the interpolated spectrum for that case.  Otherwise, interpolation is 
        done across all spectra and there is no output.

        Parameters
        ----------
        iwavelength : ndarray, optional
            Wavelengths onto which the spectra should be interpolated.
        """

        # -- set the interpolation wavelengths
        if iwavelength is None:
            self.iwavelength = self.wavelength.copy()
            self.irows = rows
            return

        # -- interpolate only one spectrum if desired
        if ltype:
            if not example:
                print("NOAA: must set desired ligting type example!!!")
                return

            try:
                leind = [i for i,(j,k) in enumerate(self.row_names) if 
                         (j==ltype) and (k==example)][0]
            except:
                print("NOAA: {0} {1} not found!!!".format(ltype,example))
                return

            return np.interp(iwavelength,self.wavelength,self.rows[leind])

        # -- interpolate over all spectra
        print("NOAA: interpolating all spectra at " 
              "{0} wavelengths".format(iwavelength.size))

        self.iwavelength = iwavelength
        self.irows       = np.array([np.interp(self.iwavelength,
                                               self.wavelength,
                                               i) for i in self.rows])

        return


    def remove_correlated(self):
        """
        For spectra which are highly correlated, this function chooses the 
        first example.
        """

        # -- set the good indices and select
        gind = np.array([0,3,7,10,11,12,16,19,20,24,28,29,30,38,39,41,42])

        self.rows      = self.rows[gind]
        self.row_names = self.row_names[gind]

        try:
            self.irows = self.irows[gind]
        except:
            pass

        return



    def binarize(self, sigma=None, interpolated=False, smooth=False):
        """
        Convert spectra to boolean values at each wavelengtqh.

        The procedure estimates the noise by taking the standard
        deviation of the derivative spectrum and dividing by sqrt(2).
        The zero-point offset for each spectrum is estimated as the
        mean of the first 10 wavelengths (empirically seen to be
        "flat" for most spectra) and is removed.  Resultant points
        >5sigma [default] are given a value of True.

        Parameters
        ----------
        sigma : float, optional
            Sets the threshold, above which the wavelength is considered to 
            have flux.

        interpolated: bool, optional
            If True, binarize the interpolated spectra.
        """

        dat = self.rows.T if not interpolated else self.irows.T

        if smooth:
            dat[:] = gf(dat,[smooth,0])

        if sigma:
            # -- estimate the noise and zero point for each spectrum
            print("BINARIZE: estimating noise level and zero-point...")
            sig = (dat[1:]-dat[:-1]).std(0)/np.sqrt(2.0)
            zer = dat[:10].mean(0)

            # -- converting to binary
            print("BINARIZE: converting spectra to boolean...")
            self.brows = ((dat-zer)>(sigma*sig)).T.copy()
        else:
            # -- binarize by comparison with mean
            self.brows = (dat > dat.mean(0)).T

        return


    def auto_correlate(self, interpolation=False):
        """
        Calculate the correlation among NOAA spectra.


        Parameters
        ----------
        interpolation : bool, optional
            If True, use interpolated spectra

        Returns
        -------
         : ndarray
            The correlation matrix of NOAA spectra
        """

        # -- Mean-subtract and normalize the data
        specs  = (self.rows if not interpolation else self.irows).T.copy()
        specs -= specs.mean(0)
        specs /= specs.std(0)

        return np.dot(specs.T,specs)/float(specs.shape[0])
