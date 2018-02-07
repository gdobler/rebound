#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os

class SourceClassifier(object):

	def __init__(self):
		# load features
		self.hsi = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','gow_stack_clip_2018.npy'))
		self.lisas = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','spectra_lisas.npy'))

		self.nosrc_load = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','hsi_nosrc_labels.npy'))
		self.src_load = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','hsi_src_labels.npy'))
		self.seed = np.random.seed(32)

		# load labeled pixel
	def build_labels(self):
		nosrc = np.empty((self.nosrc_load.shape[0],self.nosrc_load.shape[1]+1))
		src = np.empty((self.src_load.shape[0],self.src_load.shape[1]+1))

		nosrc[:,:-1] = self.nosrc_load
		src[:,:-1] = self.src_load

		nosrc[:,-1] = 0
		src[:,-1] = 1

		self.y = np.concatenate([src, nosrc], axis=0).round().astype(int)

	def build_features(self,bin_size=8):
		hsi_feat = self.hsi[:,self.y[:,1],self.y[:,0]].T
		lisas_feat = self.lisas[:,self.y[:,1],self.y[:,0]].T

		# reduce dimensions through binning
		hsi_binned = hsi_feat.reshape(hsi_feat.shape[0],hsi_feat.shape[1]//bin_size,bin_size).mean(2)
		lisas_binned = lisas_feat.reshape(lisas_feat.shape[0],lisas_feat.shape[1]//bin_size,bin_size).mean(2)

		self.x = np.append(hsi_binned, lisas_binned, axis=1)

	def split_data(self):
		# merged data
		merged = np.empty((self.x.shape[0],self.x.shape[1]+3))

		merged[:,:3] = self.y
		merged[:,3:] = self.x

		np.random.shuffle(merged)

		# 75/25 split
		cut = (merged.shape[0]//4)*3

		self.train =  merged[:cut,:]
		self.test = merged[cut:,:]

		# # i,j coordinates
		self.train_loc = self.train[:,:2]
		self.test_loc = self.test[:,:2]

		# # target variable
		self.train_y = self.train[:,2]
		self.test_y = self.test[:,2]

		# # features
		self.train_x = self.train[:,3:]
		self.test_x = self.test[:,3:]

	def scale_data(self, arr):
		return data / data.max()


