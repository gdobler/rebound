#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import settings
import numpy as np
from scipy import ndimage as nd
from sklearn import cluster
from sklearn import mixture
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

def scale_src(src_dict, gf=10):
	'''
	Scale and smooth the spectra to preprocess for clustering.

	Parameters:
	-----------
	src_dict : dict type
		Takes source dictionary from utils.mean_spectra. (keys=source labels, values = spectra)

	gf : int (default 10)
		If not None-type, performs gaussian filter over spectra axis with sigma set to gf

	Returns:
	--------
	A tuple containing 
		1) 2-d array of raw spectra (nsrcs x nwavs), 
		2) the standardized (and smoothed) array (nsrcs x nwavs)

	'''

	src_list = []
	for i in sorted(src_dict.keys()):
		src_list.append(src_dict[i].reshape(src_dict[i].shape[0], 1))

	src = np.concatenate(src_list, axis=1).T

	src_norm = (src - np.median(src, axis=1, keepdims=True))*1.0 / np.std(src, axis=1, keepdims=True)

	if gf is not None:
		src_norm = nd.gaussian_filter(src_norm, (0, gf))

	return src, src_norm

def km_cluster(scaled_srcs, n_clust=8):
	'''
	ADD DOCS!
	'''
	km = cluster.KMeans(n_clusters = n_clust)

	return km.fit_predict(scaled_srcs)

def gmm_cluster(scaled_srcs, n_clust=8, cv_type = 'full'):
	'''
	ADD DOCS!
	'''
	gmm = mixture.GaussianMixture(n_components=n_clust, covariance_type = cv_type)

	gmm.fit(scaled_srcs)

	return gmm.predict(scaled_srcs)


def dbscan_cluster(scaled_srcs, eps = 0.5, min_samples = 5):
	'''
	ADD DOCS!

	to print individ clusters, take plot(normed_srcs[model==<desired cluster >, :].T)
	'''

	db = cluster.DBSCAN(eps=eps, min_samples=min_samples)

	return db.fit_predict(scaled_srcs)


def model_eval(scaled_srcs, n_components_range= range(5,15)):
	'''
	Plots BIC score for GMM based on iterations through # components and covariance type.
	'''

	lowest_bic = np.infty
	bic = []
	cv_types = ['spherical', 'tied', 'diag', 'full']
	for cv_type in cv_types:
	    for n_components in n_components_range:

	        # Fit a Gaussian mixture with EM
	        gmm = mixture.GaussianMixture(n_components=n_components,
	                                      covariance_type=cv_type)
	        gmm.fit(scaled_srcs)
	        bic.append(gmm.bic(scaled_srcs))
	        if bic[-1] < lowest_bic:
	            lowest_bic = bic[-1]
	            best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
	                              'darkorange'])
	clf = best_gmm
	bars = []

	# Plot the BIC scores
	spl = plt.subplot(1, 1, 1)
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
	    xpos = np.array(n_components_range) + .2 * (i - 2)
	    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
	                                  (i + 1) * len(n_components_range)],
	                        width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
	    .2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	plt.show()

	return