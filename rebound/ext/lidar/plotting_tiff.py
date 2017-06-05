
# coding: utf-8

import matplotlib.pyplot as plt
import gdal
from gdalconst import *

def get_raster_info(fpath):
    raster = gdal.Open(fpath, GA_ReadOnly)
    if raster is None:
        print "Open failed"
    print 'Driver: ', raster.GetDriver().ShortName,'/', raster.GetDriver().LongName
    print 'Size is ', raster.RasterXSize,'x',raster.RasterYSize,'x',raster.RasterCount
    print 'Projection is ', raster.GetProjection()
    geotransform = raster.GetGeoTransform()
    if not geotransform is None:
        print 'Origin = (',geotransform[0], ',',geotransform[3],')'
        print 'Pixel Size = (',geotransform[1], ',',geotransform[5],')'

def get_raster_band(fpath):
    raster = gdal.Open(fpath, GA_ReadOnly)
    if raster is None:
        print "Open failed"
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
