#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import settings
import numpy as np
from scipy import ndimage as nd
from sklearn import cluster

def scale_src(src_dict):
	'''
	Scale the spectra to preprocess for clustering.

	Takes source dictionary from utils.mean_spectra. (keys=source labels, values = spectra)
	'''

	src_list = []
	for i in sorted(src_dict.keys()):
		src_list.append(src_dict[i].reshape(src_dict[i].shape[0], 1))

	src = np.concatenate(src_list, axis=1)

	return (src - np.median(src, axis=1, keepdims=True))*1.0 / np.std(src, axis=1, keepdims=True)

def cluster_src(normed_srcs, n_clust=8):
	'''
	ADD DOCS!
	'''
	km = cluster.KMeans(n_clusters = n_clust)

	return km.fit_predict(normed_srcs)

# to print this, take plot(normed_srcs[model==<desired cluster >, :].T)