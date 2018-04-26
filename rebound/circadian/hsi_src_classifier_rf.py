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

	def __init__(self, inpath=os.path.join(os.environ['REBOUND_WRITE'],'circadian')):
		# load features

		self.inpath = inpath

		nosrc = []
		src = []
		for i in os.listdir(self.inpath):
			if i[:16] == 'hsi_nosrc_labels':
				nosrc.append(np.load(os.path.join(self.inpath,i)))

			elif i[:14] == 'hsi_src_labels':
				src.append(np.load(os.path.join(self.inpath,i)))

		self.nosrc = np.concatenate(nosrc, axis=0).round().astype(int)
		self.src = np.concatenate(src, axis=0).round().astype(int)
		
		# # remove duplicate pixels
		# self.nosrc = np.array(list(set([(nosrc[i][0],nosrc[i][1]) for i in range(nosrc.shape[0])])))
		# self.src = np.array(list(set([(nosrc[i][0],nosrc[i][1]) for i in range(src.shape[0])])))

		# # balance classes
		# level_cut = min(self.nosrc.shape[0],self.src.shape[0])
		# self.nosrc = self.nosrc[:level_cut,:]
		# self.src = self.src[:level_cut,:]

		self.hsi = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','gow_stack_clip_2018.npy'))
		self.lisas = np.load(os.path.join(os.environ['REBOUND_WRITE'],'circadian','spectra_lisas.npy'))

		self.seed = np.random.seed(32)

		# load labeled pixel
	def build_labels(self):
		nosrc = np.empty((self.nosrc.shape[0],self.nosrc.shape[1]+1))
		src = np.empty((self.src.shape[0],self.src.shape[1]+1))

		nosrc[:,:-1] = self.nosrc
		src[:,:-1] = self.src

		nosrc[:,-1] = 0
		src[:,-1] = 1

		self.y = np.concatenate([src, nosrc], axis=0).astype(int)

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
		'''
		assumes shape n_samples x n_features
		default feature range is [0,1] but can be fit to training data min/max
		'''
		arr_mu = arr.mean(axis=0)
		arr_std = arr.std(axis=0)
		arr_scaled = (arr - arr_mu)*1.0 / (arr_std + (arr_std==0))

		return arr_scaled, arr_mu, arr_std

	def balanced_f(yhat, ytrue, class1=1.0):
		prec = ((yhat == class1) & (ytrue == class1)).sum() * 1.0 / (yhat == class1).sum()

		rec = ((yhat == class1) & (ytrue == class1)).sum() * 1.0 / (ytrue == class1).sum()

		return 2 * ((prec * rec) / (prec + rec))

	def run_svc(self, train_x, train_y, test_x, test_y, penalty=1.0, knl='rbf', deg=3, gm='auto'):
		sv = SVC(C=penalty, kernel=knl, degree=deg, gamma=gm)
		model = sv.fit(train_x,train_y)
		results = sv.predict(test_x)

		accuracy = (results == test_y).sum()*1.0 / test_y.shape[0]

		# f1 = self.balanced_f(results, test_y)
		f1 = f1_score(test_y, results)

		return results, accuracy, f1

	def run_rf(self, train_x, train_y, test_x, test_y, num_trees):
		rf = RandomForestClassifier(n_estimators=num_trees)
		model = rf.fit(train_x, train_y)
		results = rf.predict(test_x)

		accuracy = (results == test_y).sum()*1.0 / test_y.shape[0]

		f1 = f1_score(test_y, results)

		return results, accuracy, f1

	def kfolds(self, k, penalty=1.0, knl='rbf', deg=3, gm='auto', num_trees=None):
		merged = np.empty((self.train_x.shape[0],self.train_x.shape[1]+1))
		merged[:,:-1] = self.train_x
		merged[:,-1] = self.train_y

		np.random.shuffle(merged)

		cuts = np.linspace(0, merged.shape[0]-1,k+1).astype(int)

		kdict = {}
		k_results = {}

		for i in range(k):
			kdict[i] = merged[cuts[i]:cuts[i+1],:]

		for k,v in kdict.items():
			ktest = v
			ktrain = np.concatenate([j for i,j in kdict.items() if i != k], axis=0)

			trx = ktrain[:,:-1]
			tr_y = ktrain[:,-1]

			tex = ktest[:,:-1]
			tey = ktest[:,-1]

			k_results[k] = (trx, tr_y, tex, tey)

			trx_scld= self.scale_data(trx)[0]
			tex_scld = self.scale_data(tex)[0]

			if num_trees is None:
				res,acc,f1 = self.run_svc(trx_scld, tr_y, tex_scld, tey, penalty, knl, deg, gm)
			else:
				res,acc,f1 = self.run_rf(trx_scld, tr_y, tex_scld, tey, num_trees=num_trees)
				
			k_results[k] = (res, acc,f1)

		return k_results

	def build_dataset(self):
		self.build_labels()
		self.build_features()
		self.split_data()

	def get_cv_metrics(self,k_results):
		cv_acc = np.mean([v[1] for k,v in k_results.items()])
		cv_f1 = np.nanmean([v[2] for k,v in k_results.items()])

		return cv_acc, cv_f1

	def svc_model_eval(self, k, clog=10, glog=3, kernel_eval=False, method='f1'):
		kernels = ['linear', 'rbf', 'poly']
		poly_d = [3,4,5,6]
		c_range = np.logspace(-2, clog, 11)
		g_range = np.logspace(-9, glog, 11)


		def run_params(self, k, knl, deg, method):
			'''
			Grid for accuracy or f1 score
			'''
			param_grid = np.empty((len(c_range),len(g_range)))

			for c in range(len(c_range)):
				for g in range(len(g_range)):
					model_out = self.kfolds(k=k, penalty=c_range[c], knl=knl, deg=deg, gm=g_range[g])
					
					if method=='f1':
						param_grid[c,g] = self.get_cv_metrics(model_out)[1]
					
					elif method=='acc':
						param_grid[c,g] = self.get_cv_metrics(model_out)[0] 

			return param_grid

		if kernel_eval:
			pass
		else:
			p_grid = run_params(self, k=k, knl=kernels[1], deg=poly_d[0],method=method)

		return p_grid

		# imshow(pgrid, interpolation='nearest', cmap=cm.hot)
	def final_test(self, b_size=8, penalty=2.5188, knl='rbf', deg=3, gm=1.5e5):

		def bin_data(data, bs):
			start_idx = 0
			end_idx = bs
		
			binned = np.empty((data.shape[0]//bs, data.shape[1], data.shape[2]))
			for i in range(0, data.shape[0]//bs):
				binned[i, :, :] = data[start_idx:end_idx, : , :].mean(0)
				start_idx += bs
				end_idx += bs

			binned = binned.transpose(1, 2, 0)

			binned = binned.reshape(-1, binned.shape[-1])

			return binned

		train_x = self.scale_data(self.train_x)[0]

		hsi_bin = bin_data(self.hsi, bs=b_size)
		lisas_bin = bin_data(self.lisas, bs=b_size)
		
		self.final_test_x = self.scale_data(np.append(hsi_bin, lisas_bin, axis=1))[0]

		sv = SVC(C=penalty, kernel=knl, degree=deg, gamma=gm)
		model = sv.fit(train_x, self.train_y)
		self.results = sv.predict(self.final_test_x)


