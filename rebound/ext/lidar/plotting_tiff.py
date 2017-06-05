#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gdal
import matplotlib.pyplot as plt
from gdalconst import *

def get_raster_info(fpath):
    """
    Get info from TIF raster

    Parameters
    ----------
    fpath : str
        Full filname (including path) for the TIF raster.
    """

    # -- open the TIF raster
    try:
        raster = gdal.Open(fpath, GA_ReadOnly)
    except:
        print "Open failed"
        return

    # -- alert the user of useful info
    print('Driver: {0}/{1}'.format(raster.GetDriver().ShortName,  
                                   raster.GetDriver().LongName))
    print('Size is {0}x{1}x{2}'.format(raster.RasterXSize, 
                                       raster.RasterYSize, raster.RasterCount))
    print('Projection is {0}'.format(raster.GetProjection()))

    # -- print pixel-space to real-space conversion factors
    geotransform = raster.GetGeoTransform()
    if geotransform is not None:
        print('Origin = ({0},{1})'.format(geotransform[0], geotransform[3]))
        print('Pixel Size = ({0},{1})'.format(geotransform[1], 
                                              geotransform[5]))


def get_raster_band(fpath):
    """
    Get the elevation for a TIF raster.

    Parameters
    ----------
    fpath : str
        Full filname (including path) for the TIF raster.
    """

    # -- read the TIF raster
    try:
        raster = gdal.Open(fpath, GA_ReadOnly)
    except:
        print "Open failed"


    # -- get the data
    band = raster.GetRasterBand(1)

    print 'Band Type=',gdal.GetDataTypeName(band.DataType)
    min = band.GetMinimum()
    max = band.GetMaximum()
    if min is None or max is None:
        (min,max) = band.ComputeRasterMinMax(1)
    print 'Min=%.3f, Max=%.3f' % (min,max)
    if band.GetOverviewCount() > 0:
        print 'Band has ', band.GetOverviewCount(), ' overviews.'
    if not band.GetRasterColorTable() is None:
        print 'Band has a color table with ',     band.GetRasterColorTable().GetCount(), ' entries.'


def plot_raster(fpath):
    raster = gdal.Open(fpath, GA_ReadOnly)
    if raster is None:
        print "Open failed"
    raster_plot = raster.ReadAsArray()
    plt.imshow(raster_plot, vmin = 0)
    plt.show()
