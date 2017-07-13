#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils import read_raw
from hsi_utils import read_hyper
import scipy.ndimage.measurements as spm


# -- get file names for broadband and hsi
droot  = os.environ["REBOUND_DATA"]
bbname = os.path.join(droot, "bb", "2017", "07", "06", "stack_bb_0037777.raw")
hsname = os.path.join(droot, "slow_hsi_scans", "night_052617_00007.raw")


# -- read in the images
try:
    img_hs
except:
    img_bb = read_raw(bbname)
    img_hs = read_hyper(hsname).data[350].copy()

# sources in broadband
rr_bb = np.array([821.88822972243372, 1686.51, 1212.2626788969724])
cc_bb = np.array([3313.3156852608604, 940.936, 3543.3720854930834])


#sources in hyperspectral
rr_hsi = np.array([852.906, 1036.47, 931.829])
cc_hsi = np.array([2065.47, 1589.24, 2112.66])


#distance between broadband sources 
dist_bb = np.sqrt((rr_bb[:, np.newaxis] - rr_bb)**2 + 
                  (cc_bb[:, np.newaxis] - cc_bb)**2)

bb1 = dist_bb[0, 1]
bb2 = dist_bb[0, 2]
bb3 = dist_bb[1, 2]
print (bb1, bb2, bb3)


#distance between hyperspectral sources 
dist_hsi = np.sqrt((rr_hsi[:, np.newaxis] - rr_hsi)**2 + 
                   (cc_hsi[:, np.newaxis] - cc_hsi)**2)

hsi1 = dist_hsi[0, 1]
hsi2 = dist_hsi[0, 2]
hsi3 = dist_hsi[1, 2]
print (hsi1, hsi2, hsi3)


# -- get mean ratio of distances
factor = (bb1/hsi1 + bb2/hsi2 + bb3/hsi3) / 3


# -- get average intensity of pixels across rgb channels
img_a = img_bb.astype(float)


# -- get medians and standard deviations of luminosity images
med = np.median(img_a)
sig = img_a.std()


# -- get the thresholded images
thr = img_a > (med + 4.0*sig)


# -- label the sources
labs = spm.label(thr)


# -- get the source sizes
lsz = spm.sum(thr, labs[0], range(1, labs[1]+1))


# -- get the positions of the sources
ind = (lsz > 200.) & (lsz < 1700.) 


# -- get center of masses for all the labelled sources in the image
rr1, cc1, = np.array(spm.center_of_mass(thr, labs[0], 
                                   np.arange(1, labs[1]+1)[ind])).T


#Calculating distances between sources of image
dist = np.sqrt((rr1[:, np.newaxis] - rr1)**2 + (cc1[:, np.newaxis] - cc1)**2)


# -- rescale for bb/hsi comparison
dist /= factor


def find(dist, dcat, buff=10):
    """
    Find potential sources with the appropriate distance  ratios
    """
    src  = np.arange(dist.shape[0])
    dcat = dcat[dcat > 0]
    for tdist in dcat:
        tind = (np.abs(dist - tdist) < buff).any(1)
        dist = dist[tind]
        src  = src[tind]

#        if tdist == dcat[-1]:
#            return src

    return src



def find_quads(dist, rr_cat, cc_cat):
    """
    Find quads of sources with the appropriate distance ratios.
    """

    # -- trim rows that do not have that distance distribution
    pts  = []

    # -- get all possible points
    allind = np.arange(dist.shape[0])
    
    dcat0 = np.sqrt(((rr_cat[0] - rr_cat)**2 + (cc_cat[0] - cc_cat)**2))
    dcat1 = np.sqrt(((rr_cat[1] - rr_cat)**2 + (cc_cat[1] - cc_cat)**2))
    dcat2 = np.sqrt(((rr_cat[2] - rr_cat)**2 + (cc_cat[2] - cc_cat)**2))
    
    p0ind = find(dist, dcat0)
    p1ind = find(dist, dcat1)
    p2ind = find(dist, dcat2)
    
    return p0ind, p1ind, p2ind


# -- get potential source indices
p0ind, p1ind, p2ind = find_quads(dist, rr_hsi, cc_hsi)


# -- compare multiple distances
dcat0 = np.sqrt(((rr_hsi[0] - rr_hsi)**2 + (cc_hsi[0] - cc_hsi)**2))
dcat1 = np.sqrt(((rr_hsi[1] - rr_hsi)**2 + (cc_hsi[1] - cc_hsi)**2))
dcat2 = np.sqrt(((rr_hsi[2] - rr_hsi)**2 + (cc_hsi[2] - cc_hsi)**2))
good01 = []
for ii in p0ind:
    for jj in p1ind:
        if np.abs(dist[ii, jj] - dcat0[1]) < 10:
            good01.append([ii, jj])
good012 = []
for ii, jj in good01:
    for kk in p2ind:
        flag02 = np.abs(dist[ii, kk] - dcat0[2]) < 10
        flag12 = np.abs(dist[jj, kk] - dcat1[2]) < 10
        if flag02 and flag12:
            good012.append([ii, jj, kk])


# -- set paired down indices
p0s, p1s, p2s = np.array(good012).T


# -- find the delta angles of the first 2 pairs
theta01 = np.arccos((rr1[p0s] - rr1[p1s]) / (dist[p0s, p1s] * factor))
theta02 = np.arccos((rr1[p0s] - rr1[p2s]) / (dist[p0s, p2s] * factor))
dtheta  = (theta01 - theta02) * 180. / np.pi


# -- get hsi distance matrix
dcatm = np.sqrt((rr_hsi[:, np.newaxis] - rr_hsi)**2 + 
                (cc_hsi[:, np.newaxis] - cc_hsi)**2)


# -- find the delta angle of thr first 2 sources of catalog
theta01_cat = np.arccos((rr_hsi[0] - rr_hsi[1]) / dcatm[0, 1])
theta02_cat = np.arccos((rr_hsi[0] - rr_hsi[2]) / dcatm[0, 2])
dtheta_cat  = (theta01_cat - theta02_cat) * 180. / np.pi


# -- compare the difference in angles
guess = np.array(good012[np.abs(dtheta - dtheta_cat).argmin()])




# -- get positions of image sources and catalog sources to know the 
#    quadrant they occupy
# rrr0, ccc0 = rr_hsi[np.array([0, 1, 2])], cc_hsi[np.array([0, 1, 2])]
# rrr1, ccc1 = rr1[guess] / factor, cc1[guess] / factor

# roff_hsi = img_hs.shape[0] / 2
# coff_hsi = img_hs.shape[1] / 2
# roff_bb  = img_bb.shape[0] / factor / 2
# coff_bb  = img_bb.shape[1] / factor / 2

# rrr0 -= roff_hsi
# ccc0 -= coff_hsi
# rrr1 -= roff_bb
# ccc1 -= coff_bb


rrr0, ccc0 = rr_hsi[np.array([0, 1, 2])], cc_hsi[np.array([0, 1, 2])]
rrr1, ccc1 = rr1[guess], cc1[guess]

roff_hsi = img_hs.shape[0] / 2
coff_hsi = img_hs.shape[1] / 2
roff_bb  = img_bb.shape[0] / 2
coff_bb  = img_bb.shape[1] / 2

rrr0 -= roff_hsi
ccc0 -= coff_hsi
rrr1 -= roff_bb
ccc1 -= coff_bb

rrr1 /= factor
ccc1 /= factor

# -- get the matrix for catalog sources
mones      = np.zeros(rrr0.size * 2 + 1)
mones[::2] = 1.0

pm         = np.zeros([rrr0.size * 2, 4])
pm[::2,0]  = rrr0
pm[1::2,0] = ccc0
pm[::2,1]  = -ccc0
pm[1::2,1] = rrr0
pm[:,2]    = mones[:-1]
pm[:,3]    = mones[1:]

# -- get the matrix for image sources
bv         = np.zeros([rrr0.size * 2])
bv[::2]    = rrr1
bv[1::2]   = ccc1

# -- calculating rotation of image 
pmTpm = np.dot(pm.T, pm)
av    = np.dot(np.linalg.inv(pmTpm), np.dot(pm.T, bv))

dr, dc = av[-2:]
dtheta = np.arctan2(av[1], av[0]) * 180. / np.pi


# -- test this on a source
rt, ct = 1295, 1817

rrt  = rt - roff_bb
cct  = ct - coff_bb
rrt /= factor
cct /= factor

rt_hsi = rrt*np.cos(dtheta * np.pi / 180.) - \
    cct*np.sin(dtheta * np.pi / 180.) - dr + roff_hsi
ct_hsi = rrt*np.sin(dtheta * np.pi / 180.) + \
    cct*np.cos(dtheta * np.pi / 180.) - dc + coff_hsi


# -- convert all bb row and col indices to hsi indices
nrow = img_bb.shape[0]
ncol = img_bb.shape[1]
rind = np.arange(nrow * ncol).reshape(nrow, ncol) // ncol
cind = np.arange(nrow * ncol).reshape(nrow, ncol) % ncol

rrv  = (rind - roff_bb).astype(float)
ccv  = (cind - coff_bb).astype(float)
rrv /= factor
ccv /= factor

rt_hsi = (rrv*np.cos(dtheta * np.pi / 180.) - 
          ccv*np.sin(dtheta * np.pi / 180.) - dr + roff_hsi) \
          .round().astype(int)
ct_hsi = (rrv*np.sin(dtheta * np.pi / 180.) + 
          ccv*np.cos(dtheta * np.pi / 180.) - dc + coff_hsi) \
          .round().astype(int)

rgb = np.zeros(list(img_hs.shape) + [3], dtype=np.uint8)


rgb[..., 0] = (2.0*(1.0*img_hs - 150)).clip(0, 200).astype(np.uint8)
rgb[rt_hsi, ct_hsi, 2] = img_bb[rind, cind]
