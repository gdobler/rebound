#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from utils import read_raw
from hsi_utils import read_hyper
import scipy.ndimage.measurements as spm


# -- read images
droot  = os.environ["REBOUND_DATA"]
bbname = os.path.join(droot, "bb", "2017", "07", "06", "stack_bb_0037777.raw")
hsname = os.path.join(droot, "slow_hsi_scans", "night_052617_00007.raw")
try:
    img_hs
except:
    img_bb = read_raw(bbname)
    img_hs = read_hyper(hsname).data[350].copy()


# -- read sources
sname = os.path.join(os.environ["REBOUND_WRITE"], "sources.csv")
srcs  = pd.read_csv(sname)


# -- extract positions
rrr0, ccc0 = srcs.hs_r.values.copy(), srcs.hs_c.values.copy()
rrr1, ccc1 = srcs.bb_r.values.copy(), srcs.bb_c.values.copy()


# -- correct for the aspect ratio of the HSI pixels
asp0   = 0
asp1   = 2
rat_bb = (rrr1[asp0] - rrr1[asp1]) / (ccc1[asp1] - ccc1[asp0])
rat_hs = (rrr0[asp1] - rrr0[asp0]) / (ccc0[asp0] - ccc0[asp1])
fac_hs =  rat_bb / rat_hs

print("HSI aspect ratio = {0}".format(fac_hs))
    
ccc0 /= fac_hs


# -- get the mean scaling factor between distances
dist_hs = np.sqrt((rrr0[:, np.newaxis] - rrr0)**2 + 
                  (ccc0[:, np.newaxis] - ccc0)**2)
dist_bb = np.sqrt((rrr1[:, np.newaxis] - rrr1)**2 + 
                  (ccc1[:, np.newaxis] - ccc1)**2)
factor  = (dist_bb[dist_bb > 0]/dist_hs[dist_bb>0]).mean()


# -- fit registration solution
roff_hsi = img_hs.shape[0] / 2
coff_hsi = img_hs.shape[1] / 2 / fac_hs
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


# -- convert all bb row and col indices to hsi indices
nrow = img_bb.shape[0]
ncol = img_bb.shape[1]
rind = np.arange(nrow * ncol).reshape(nrow, ncol) // ncol
cind = np.arange(nrow * ncol).reshape(nrow, ncol) % ncol

rrv  = (rind - roff_bb).astype(float)
ccv  = (cind - coff_bb).astype(float)
rrv /= factor
ccv /= factor

rt_hsi = (rrv * np.cos(-dtheta * np.pi / 180.) -
           ccv * np.sin(-dtheta * np.pi / 180.) - dr + roff_hsi) \
           .round().astype(int)

ct_hsi = ((rrv * np.sin(-dtheta * np.pi / 180.) +
           ccv * np.cos(-dtheta * np.pi / 180.) - dc + coff_hsi) * fac_hs) \
           .round().astype(int)


# -- plot the result
xs = 8.0
ys = xs * float(img_hs.shape[0]) / float(img_hs.shape[1])
rgb = np.zeros(list(img_hs.shape) + [3], dtype=np.uint8)
rgb[..., 0] = (15.0*(1.0*img_hs - 150)).clip(0, 200).astype(np.uint8)
rgb[rt_hsi, ct_hsi, 2] = (10.*img_bb[rind, cind]).clip(0, 255).astype(np.uint8)

fig, ax = plt.subplots(figsize=(xs, ys))
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
im = ax.imshow(rgb)
fig.canvas.draw()


# -- print the input hsi source locations and the derived
rrrt = srcs.bb_r.values
ccct = srcs.bb_c.values

rrv  = (rrrt - roff_bb).astype(float)
ccv  = (ccct - coff_bb).astype(float)
rrv /= factor
ccv /= factor

rv_hsi = (rrv * np.cos(-dtheta * np.pi / 180.) -
           ccv * np.sin(-dtheta * np.pi / 180.) - dr + roff_hsi)
cv_hsi = (rrv * np.sin(-dtheta * np.pi / 180.) +
          ccv * np.cos(-dtheta * np.pi / 180.) - dc + coff_hsi) * fac_hs

print("HSI_r in  = {0}\nHSI_r out = {1}".format(srcs.hs_r.values, rv_hsi))
print("")
print("HSI_c in  = {0}\nHSI_c out = {1}".format(srcs.hs_c.values, cv_hsi))
