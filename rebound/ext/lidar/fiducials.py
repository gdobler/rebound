#!/usr/bin/env python

import numpy as np

ldr_fid = np.array()
img_fid = np.array([[2144, 656], # 155 Dean St
		    [1096, 351], # 370 4th Ave
		    [1446, 967], # 198 Bond St
		    [1089, 3439], # 126 13th St
		    [1542, 4005], # 241 Hoyt St
		    [1936, 2910], # 175 Hoyt St
		    [1668, 1984]])# 417 Baltic St

img_dim = np.array([3072, 4096])

def center_img_fid(img_dim, img_fid):

	'''Center image fiducial points using image dimensions'''
	
	for i in range(np.shape(img_fid)[0]):
		img_fid[i, 0] = img_fid[i, 0] - (img_dim[0] / 2)
		img_fid[i, 1] = img_fid[i, 1] - (img_dim[1] / 2)
	return img_fid
