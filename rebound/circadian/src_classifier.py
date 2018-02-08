#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import f1_score

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

	def scale_data(self, arr, feat_range=[0,1]):
		'''
		assumes shape n_samples x n_features
		default feature range is [0,1] but can be fit to training data min/max
		'''
		arr_std = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))

		arr_scaled = arr_std * (feat_range[1] - feat_range[0]) + feat_range[0]

		return arr_std, arr_scaled

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

	def kfolds(self, k, penalty=1.0, knl='rbf', deg=3, gm='auto'):
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

			# trx_scld= self.scale_data(trx)[1]
			# tex_scld = self.scale_data(tex)[1]
			trx_scld= trx.copy()
			tex_scld = tex.copy()

			res,acc,f1 = self.run_svc(trx_scld, tr_y, tex_scld, tey, penalty, knl, deg, gm)
			k_results[k] = (res, acc,f1)
			# k_results[k] = acc

		return k_results

	def build_dataset(self):
		self.build_labels()
		self.build_features()
		self.split_data()

	def get_cv_metrics(self,k_results):
		cv_acc = np.mean([v[1] for k,v in k_results.items()])
		cv_f1 = np.nanmean([v[2] for k,v in k_results.items()])

		return cv_acc, cv_f1

	def model_eval(self, k, clog=10, glog=3, kernel_eval=False, method='f1'):
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



