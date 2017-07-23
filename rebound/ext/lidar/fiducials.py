#!/usr/bin/env python

import numpy as np

def get_img_dims(image):
	'''Get image dimensions: x = column pixels, y = row pixels'''
	if image == 'bk':
		dims = np.array([4096, 3072])
	else:
		print "Not a valid image"
		return
	return dims

def get_img_fid(image, center = False):
	'''Get image pixel location of fiducial points, columns first'''
	if image == 'bk':
		img_fid = np.array([[656, 2144],   # 155 Dean St front-left corner
		   		    [351, 1096],   # 370 4th Ave
		   		    [967, 1446],   # 198 Bond St
		   		    [3439, 1089],  # 126 13th St
		   		    [4005, 1542],  # 241 Hoyt St
		   		    [2910, 1936],  # 175 Hoyt St
		   		    [1984, 1668],  # 417 Baltic St
				    [1894, 2165],  # 155 Dean St back-right corner
				    [786, 2482],   # 97 Hoyt St
				    [2934, 2483]]) # 160 Schermerhorn St

	else:
		print "Not a valid image"
		return

	dims = get_img_dims(image)

	if center == True:
		img_fid = center_img_fid(dims, img_fid)

	return img_fid

def get_ldr_fid(image):
	'''Get x,y,z of fiducial points in LiDAR'''
	if image == 'bk':
		ldr_fid = np.array([[9.87858446e+05, 1.89366284e+05, 4.25000000e+01],
      				    [9.87851166e+05, 1.84582434e+05, 5.80000000e+01],
      				    [9.87753166e+05, 1.88285434e+05, 8.00000000e+01],
      				    [9.86039166e+05, 1.83084434e+05, 3.20000000e+01],
      				    [9.87241166e+05, 1.88546434e+05, 6.50000000e+01],
      				    [9.87414166e+05, 1.88864434e+05, 4.90000000e+01],
      				    [9.87556166e+05, 1.88604434e+05, 6.60000000e+01],
      				    [9.87671816e+05, 1.89426084e+05, 4.25000000e+01],
      				    [9.87855866e+05, 1.89793684e+05, 3.98000000e+01],
      				    [9.87694656e+05, 1.90460144e+05, 7.92000000e+01]])

def center_img_fid(dims, img_fid):

	'''Center fiducial points using image dimensions'''
	center_fid = np.zeros([[img_fid[0], img_fid[1]]) 	
	
	for i in range(np.shape(img_fid)[0]):
		center_fid[i, 0] = img_fid[i, 0] - (dims[0] / 2)
		center_fid[i, 1] = img_fid[i, 1] - (dims[1] / 2)
	
	return center_fid

