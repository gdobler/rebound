#!/usr/bin/env python

import numpy as np

img_fid = np.array([[2144, 656],   # 155 Dean St
		    [1096, 351],   # 370 4th Ave
		    [1446, 967],   # 198 Bond St
		    [1089, 3439],  # 126 13th St
		    [1542, 4005],  # 241 Hoyt St
		    [1936, 2910],  # 175 Hoyt St
		    [1668, 1984]]) # 417 Baltic St

ldr_fid = np.array([[9.87856166e+05, 1.89364434e+05, 4.60000000e+01],
      		    [9.87851166e+05, 1.84582434e+05, 5.80000000e+01],
        	    [9.87753166e+05, 1.88285434e+05, 8.00000000e+01],
      		    [9.86039166e+05, 1.83084434e+05, 3.20000000e+01],
      		    [9.87241166e+05, 1.88546434e+05, 6.50000000e+01],
      		    [9.87414166e+05, 1.88864434e+05, 4.90000000e+01],
      		    [9.87556166e+05, 1.88604434e+05, 6.60000000e+01]])

img_dim = np.array([3072, 4096])

def center_img_fid(img_dim, img_fid):

	'''Center image fiducial points using image dimensions'''
	
	for i in range(np.shape(img_fid)[0]):
		img_fid[i, 0] = img_fid[i, 0] - (img_dim[0] / 2)
		img_fid[i, 1] = img_fid[i, 1] - (img_dim[1] / 2)
	return img_fid

def run(img_fid, ldr_fid, num_iter, params=None):

	''' Run the optimization routine for the given image '''

	if params == None:
		print "Choosing 1MT UO as initial guess\n"
		params = np.array([1.56926535e+00, -1.20789690e-01,\
			-3.05255789e-03, 9.87920425e+05, 1.91912958e+05, \
			3.85333237e+02, -1.10001068e+04]) 

	xyz_s = ldr_fid
	xy_t = img_fid

	min_score = 100000000000000

	for i in range(0, num_iter):
		result = call(params,xyz_s,xy_t)
		print "params, score", result.x, result.fun
#		print('doing it {0} {1}'.format(i,result.fun))
#		import pdb; pdb.set_trace()
		if (result.fun < min_score):# and (result.x[3] < 980491):
			min_score = result.fun
			params = result.x

	return [min_score, params]
