#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import geopandas as gp
from shapely.geometry import Point

# -- load the x/y grid
xgrid = np.load(os.path.join(os.environ["REBOUND_WRITE"], 'xgrid.npy'))
ygrid = np.load(os.path.join(os.environ["REBOUND_WRITE"], 'ygrid.npy'))

# -- read in the mappluto data
print "reading data and extracting x,y of lot centroids"
bk = gp.GeoDataFrame.from_file(os.path.join(os.environ["REBOUND_DATA"], "BKMapPLUTO.shp"))
bk_cenx = np.array([i.x for i in bk['geometry'].centroid])
bk_ceny = np.array([i.y for i in bk['geometry'].centroid])
bk_geo  = np.array(bk['geometry'])
bk_bbl  = np.array(bk['BBL'])

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
        ind  = ((bk_cenx-xpos)**2+(bk_ceny-ypos)**2)<rad2
        for geo,bbl in zip(bk_geo[ind],bk_bbl[ind]):
            if geo.contains(pnt):
                bblgrid[ii,jj] = bbl
                continue

# -- write to file
np.save(os.path.join(os.environ["REBOUND_WRITE"], 'bblgrid.npy'), bblgrid)

