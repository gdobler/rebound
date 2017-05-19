#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
import util_test

def data_extract(data_in,shape,hyper,verbose):
	"""
	Returns tuple where tuple[0] = output from read_raw function in util_test.py 
	and tuple[1] = reshaped dimensions of raw_data
	"""
	if not hyper:
		return util_test.read_raw(data_in,shape,hyper,verbose),(shape[0],shape[1])

def compress(input_fpath, shape, output_folder, compress=True, compress_ratio=0, hyper=False,verbose=True):
	"""
	Loads input raw image data into an HDF5 file object and gives option to compress
	Current setup requires that you create a data output folder with two sub-folders: 'hyperspectral/' and 'broadband/'
	"""
	file_name = input_fpath.split('/')[-1]
	file_name = file_name.split('.')[0]+".hdf5"
	if hyper:
		output_folder = output_folder+'/hyperspectral/'
	else:
		output_folder = output_folder+'/broadband/'

	file = h5py.File(output_folder+file_name,'w')

	data_in = data_extract(input_fpath, shape, hyper, verbose)

	if compress:
		dset = file.create_dataset(file_name, data_in[1], dtype=np.uint8,
			compression='gzip', compression_opts=compress_ratio)
		dset.attrs['compression_value'] = compress_ratio

	else:
		dset = file.create_dataset(file_name, data_in[1], dtype=np.uint8)

	dset.attrs['shape'] = data_in[1]
	dset[...] = data_in[0]

	file.close()