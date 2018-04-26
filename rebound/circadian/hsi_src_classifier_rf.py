#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

class RFClassifier(object):
	'''
	This classifier trains a random forest ensemble on hyper-sepctral image scans (HSI). This scan is a stacked datacube
	of all HSI scans in the observation period (limited to the Gowanus field of view).

	THis method uses previously engineered features along 2 dimensions for a given observration (pixel): 
	1. HSI intensity per channel (n=848)
	2. The weight co-efficient produced by calculating a pixel's spatial autocorrelation 
	with its neighbors (see spatial_auto.py) (n=848)

	Both dimensions are binned and flattened to create a feature vector of 212 dimensions 
	(106 HSI values + 106 spatial wghts)

	The observations (pixels) used for training and testing were previously selected and labeled manually (by sight). 
	(n=3001, 1380 source, 1621 non-source)
	
	Data are split into text/train sets and whitened.

	A RF classifier is used to classify pixels as being "source" or "non-source" and both accuracy and f1 score metrics 
	are produced.

	Finally, the trained classifier is run on the entire 300 x 800 pixel Gowanus HSI scan. 

	This classifier could also be run on the full 1600 x 3195 scan if the spatial autocorrelation engineering is performed
	on the full scan.
	'''

	def __init__(self, spatial=True, bin_size = 8):
		'''
		spatial == True for using spatial and HSi features, false for temporal
		bin size = number of features to group per bin for dimension reduction
		'''

		self.spatial = spatial
		self.bin_size = bin_size
		self.seed = np.random.seed(32)

	# load labeled pixels
	def load_labels(self, fpath):

		# nobs x features (x,y,label) -> 3001 x 3
		self.labels = np.load(fpath)

	def load_features(self):
		
		if self.spatial:
			# nwavs/wgts x nrows x ncols -> 848 x 300 x 800
			self.hsi = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','gow_stack_clip_2018.npy'))
			self.spatial_wgt = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','spectra_lisas.npy'))

		else:
			# for future adaption to use temporal features...
			pass


	def build_training_set(self):

		if self.spatial:
			hsi_feat = self.hsi[:,self.labels[:,0],self.labels[:,1]].T
			sa_feat = self.spatial_wgt[:,self.labels[:,0],self.labels[:,1]].T

			# reduce dimensions through binning
			hsi_binned = hsi_feat.reshape(hsi_feat.shape[0],hsi_feat.shape[1]//self.bin_size,self.bin_size).mean(2)
			sa_binned = sa_feat.reshape(sa_feat.shape[0],sa_feat.shape[1]//self.bin_size,self.bin_size).mean(2)

			self.training_feat = np.append(hsi_binned, sa_binned, axis=1)

		else:
			pass


	def split_data(self):
		# merged data
		data = np.empty((self.training_feat.shape[0],self.training_feat.shape[1]+1))

		data[:,:-1] = self.training_feat
		data[:,-1] = self.labels[:,2]

		np.random.shuffle(data)

		# 75/25 split
		cut = (data.shape[0]//4)*3

		self.train =  data[:cut,:]
		self.test = data[cut:,:]

		# for data whitening, create params from training data
		self.mu = self.train[:,:-1].mean(axis=0)
		self.std = self.train[:,:-1].std(axis=0)


	def whiten_data(self, data):
		'''
		assumes shape n_samples x n_features
		default feature range is [0,1] but can be fit to training data min/max
		Note: to prevent over-fitting, use the training data parameters for whitening test data
		'''
		return (data - self.mu)*1.0 / (self.std + (self.std==0))


	def build(self, fpath = os.path.join(os.environ['HOME_UO'],'circadian','hsi_gow_training_labels.npy')):

		self.load_labels(fpath=fpath)
		self.load_features()
		self.build_training_set()
		self.split_data()

		self.train_x = self.whiten_data(self.train[:,:-1])
		self.train_y = self.train[:,-1]
		self.test_x = self.whiten_data(self.test[:,:-1])
		self.test_y = self.test[:,-1]

	def train_rf(self, num_trees, verbose = 0):
		self.rf = RandomForestClassifier(n_estimators=num_trees, verbose = verbose)
		self.model = self.rf.fit(self.train_x, self.train_y)

	def test_rf(self):
		self.results = self.rf.predict(self.test_x)
		self.acc = (self.results == self.test_y).sum()*1.0 / self.test_y.shape[0]
		self.f1 = f1_score(self.test_y, self.results)

		print('Accuracy on test data: {}\nF1 score on test data:{}'.format(self.acc, self.f1))

	def classify(self):
		if self.spatial:

			hsi_feat = self.hsi.reshape(self.hsi.shape[0],self.hsi.shape[1] * self.hsi.shape[2]).T
			sa_feat = self.spatial_wgt.reshape(self.spatial_wgt.shape[0],self.spatial_wgt.shape[1] * self.spatial_wgt.shape[2]).T

			# reduce dimensions through binning
			hsi_binned = hsi_feat.reshape(hsi_feat.shape[0],hsi_feat.shape[1]//self.bin_size,self.bin_size).mean(2)
			sa_binned = sa_feat.reshape(sa_feat.shape[0],sa_feat.shape[1]//self.bin_size,self.bin_size).mean(2)

			self.all_feat = np.append(hsi_binned, sa_binned, axis=1)

		else:
			pass

		self.input_feat = self.whiten_data(self.all_feat)

		self.predicted = self.rf.predict(self.input_feat).reshape(self.hsi.shape[1], self.hsi.shape[2])



	def plot(self):
		# plot results
		pass

