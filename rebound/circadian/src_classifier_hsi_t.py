#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import utils
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

# optimum c = 2.5118
# optimum gamma = 1.5e5

class SourceClassifier(object):

	def __init__(self, lpath=os.path.join(os.environ['HOME_UO'],'circadian','hsi_training_labels.npy')):
		# load features

		self.labels = np.load(lpath)

		self.nights = [('07','29'),('07','30'),('08','01'),('08','02'),('08','03'),('08','04'),('08','05'),
          ('08','06'),('08','07'),('08','08'),('08','09'),('08','10'),('08','11'),('08','12'),
          ('08','13'),('08','14'),('08','15'),('08','16'),('08','17'),('08','18'),('08','19')]

		self.seed = np.random.seed(32)

	def build_features(self,bin_size=8):

		n_nights = 1

		features = np.empty((self.labels.shape[0], n_nights*80*(848/bin_size)))

		dpath = os.path.join(os.environ['REBOUND_DATA'],'hsi','2017')

		bin_edges = np.linspace(0,848,848/bin_size)
			
		idx = 0
		icount = 0

		for n in self.nights[:n_nights]:

			for i in sorted(os.listdir(os.path.join(dpath, n[0], n[1])))[:2]:
				if i.split('.')[-1]=='raw':

					if icount % 50 == 0:
						print('Reading {} of {}...'.format(icount, n_nights*80))

					fpath = os.path.join(dpath, n[0], n[1],i)

					hdr = utils.read_header(fpath.replace("raw", "hdr"))
					sh  = (hdr["nwav"], hdr["nrow"], hdr["ncol"])

					data = utils.read_hyper(fpath).data.copy()
					data = data.reshape(sh[2], sh[0], sh[1])[:, :, ::-1].transpose(1, 2, 0)


					self.x = data[:,self.labels[:,1], self.labels[:,0]].T




		


		# self.x = sample_features

		# reduce dimensions through binning
		# hsi_binned = hsi_feat.reshape(hsi_feat.shape[0],hsi_feat.shape[1]//bin_size,bin_size).mean(2)
		# lisas_binned = lisas_feat.reshape(lisas_feat.shape[0],lisas_feat.shape[1]//bin_size,bin_size).mean(2)

		# self.x = np.append(hsi_binned, lisas_binned, axis=1)

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

	# def balanced_f(yhat, ytrue, class1=1.0):
	# 	prec = ((yhat == class1) & (ytrue == class1)).sum() * 1.0 / (yhat == class1).sum()

	# 	rec = ((yhat == class1) & (ytrue == class1)).sum() * 1.0 / (ytrue == class1).sum()

	# 	return 2 * ((prec * rec) / (prec + rec))


	# def run_rf(self, train_x, train_y, test_x, test_y, num_trees):
	# 	rf = RandomForestClassifier(n_estimators=num_trees)
	# 	model = rf.fit(train_x, train_y)
	# 	results = rf.predict(test_x)

	# 	accuracy = (results == test_y).sum()*1.0 / test_y.shape[0]

	# 	f1 = f1_score(test_y, results)

	# 	return results, accuracy, f1

	# def kfolds(self, k, penalty=1.0, knl='rbf', deg=3, gm='auto', num_trees=None):
	# 	merged = np.empty((self.train_x.shape[0],self.train_x.shape[1]+1))
	# 	merged[:,:-1] = self.train_x
	# 	merged[:,-1] = self.train_y

	# 	np.random.shuffle(merged)

	# 	cuts = np.linspace(0, merged.shape[0]-1,k+1).astype(int)

	# 	kdict = {}
	# 	k_results = {}

	# 	for i in range(k):
	# 		kdict[i] = merged[cuts[i]:cuts[i+1],:]

	# 	for k,v in kdict.items():
	# 		ktest = v
	# 		ktrain = np.concatenate([j for i,j in kdict.items() if i != k], axis=0)

	# 		trx = ktrain[:,:-1]
	# 		tr_y = ktrain[:,-1]

	# 		tex = ktest[:,:-1]
	# 		tey = ktest[:,-1]

	# 		k_results[k] = (trx, tr_y, tex, tey)

	# 		trx_scld= self.scale_data(trx)[0]
	# 		tex_scld = self.scale_data(tex)[0]

	# 		if num_trees is None:
	# 			res,acc,f1 = self.run_svc(trx_scld, tr_y, tex_scld, tey, penalty, knl, deg, gm)
	# 		else:
	# 			res,acc,f1 = self.run_rf(trx_scld, tr_y, tex_scld, tey, num_trees=num_trees)
				
	# 		k_results[k] = (res, acc,f1)

	# 	return k_results

	# def build_dataset(self):
	# 	self.build_labels()
	# 	self.build_features()
	# 	self.split_data()

	# def get_cv_metrics(self,k_results):
	# 	cv_acc = np.mean([v[1] for k,v in k_results.items()])
	# 	cv_f1 = np.nanmean([v[2] for k,v in k_results.items()])

	# 	return cv_acc, cv_f1

	# def svc_model_eval(self, k, clog=10, glog=3, kernel_eval=False, method='f1'):
	# 	kernels = ['linear', 'rbf', 'poly']
	# 	poly_d = [3,4,5,6]
	# 	c_range = np.logspace(-2, clog, 11)
	# 	g_range = np.logspace(-9, glog, 11)


	# 	def run_params(self, k, knl, deg, method):
	# 		'''
	# 		Grid for accuracy or f1 score
	# 		'''
	# 		param_grid = np.empty((len(c_range),len(g_range)))

	# 		for c in range(len(c_range)):
	# 			for g in range(len(g_range)):
	# 				model_out = self.kfolds(k=k, penalty=c_range[c], knl=knl, deg=deg, gm=g_range[g])
					
	# 				if method=='f1':
	# 					param_grid[c,g] = self.get_cv_metrics(model_out)[1]
					
	# 				elif method=='acc':
	# 					param_grid[c,g] = self.get_cv_metrics(model_out)[0] 

	# 		return param_grid

	# 	if kernel_eval:
	# 		pass
	# 	else:
	# 		p_grid = run_params(self, k=k, knl=kernels[1], deg=poly_d[0],method=method)

	# 	return p_grid

	# 	# imshow(pgrid, interpolation='nearest', cmap=cm.hot)
	# def final_test(self, b_size=8, penalty=2.5188, knl='rbf', deg=3, gm=1.5e5):

	# 	def bin_data(data, bs):
	# 		start_idx = 0
	# 		end_idx = bs
		
	# 		binned = np.empty((data.shape[0]//bs, data.shape[1], data.shape[2]))
	# 		for i in range(0, data.shape[0]//bs):
	# 			binned[i, :, :] = data[start_idx:end_idx, : , :].mean(0)
	# 			start_idx += bs
	# 			end_idx += bs

	# 		binned = binned.transpose(1, 2, 0)

	# 		binned = binned.reshape(-1, binned.shape[-1])

	# 		return binned

	# 	train_x = self.scale_data(self.train_x)[0]

	# 	hsi_bin = bin_data(self.hsi, bs=b_size)
	# 	lisas_bin = bin_data(self.lisas, bs=b_size)
		
	# 	self.final_test_x = self.scale_data(np.append(hsi_bin, lisas_bin, axis=1))[0]

	# 	sv = SVC(C=penalty, kernel=knl, degree=deg, gamma=gm)
	# 	model = sv.fit(train_x, self.train_y)
	# 	self.results = sv.predict(self.final_test_x)


