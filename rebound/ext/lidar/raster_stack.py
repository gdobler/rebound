
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gdal
import glob
import numpy as np
from gdalconst import *

fpattern = os.path.join(os.environ['REBOUND_DATA'], "data_3d", "DA18_Raster_Files", "DA18_Raster_Files", "DA18RasterTile")
fpath = "%s*.TIF" % (fpattern)
flist = glob.glob(fpath)

origin = (0, 0)
result = np.zeros((2048, 2048))

for fname in (sorted(flist)):
    ds = gdal.Open(fname, GA_ReadOnly)
    geotransform = ds.GetGeoTransform()
    tile_origin = geotransform[0], geotransform[3]
    raster = ds.ReadAsArray()
    if tile_origin[0] == origin[0] + 2048 and tile_origin[1] == origin[1]: 
        result = np.hstack((result, raster))
    elif tile_origin[0] == origin[0] and tile_origin[1] == origin[1] - 2048: 
        result = np.vstack((result, raster))
    else:
        pass
    origin = tile_origin

#print result.shape
