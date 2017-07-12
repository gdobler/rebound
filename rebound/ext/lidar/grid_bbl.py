#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import geopandas as gp
from shapely.geometry import Point

# -- load the x/y grid
xgrid = np.load('../output/lwir_xgrid.npy')
ygrid = np.load('../output/lwir_ygrid.npy')

# -- read in the mappluto data
print "reading data and extracting x,y of lot centroids"
mn = gp.GeoDataFrame.from_file('../../data/mappluto/Manhattan/MNMapPLUTO.shp')
mn_cenx = np.array([i.x for i in mn['geometry'].centroid])
mn_ceny = np.array([i.y for i in mn['geometry'].centroid])
mn_geo  = np.array(mn['geometry'])
mn_bbl  = np.array(mn['BBL'])

# -- initialize bbl grid
bblgrid = np.zeros(xgrid.shape)

# -- loop through pixels
rad2 = 500.**2
nrow, ncol = xgrid.shape
for ii in range(nrow):
    print "row {0} of {1}\r".format(ii+1,nrow),
    sys.stdout.flush()
    for jj in range(ncol):
        xpos = xgrid[ii,jj]
        ypos = ygrid[ii,jj]
        pnt  = Point(xpos,ypos)
        ind  = ((mn_cenx-xpos)**2+(mn_ceny-ypos)**2)<rad2
        for geo,bbl in zip(mn_geo[ind],mn_bbl[ind]):
            if geo.contains(pnt):
                bblgrid[ii,jj] = bbl
                continue

# -- write to file
np.save('../output/lwir_bblgrid.npy',bblgrid)

