#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class MoranLisa(object):

    def __init__(self, img, dist_limit):
        self.img = img
        self.dthresh = np.sqrt(2*dist_limit**2)
        self.dist_limit = dist_limit
        self.rmax = self.img.shape[0]
        self.cmax = self.img.shape[1]
        print("Call .get_stats() method to generate lisa.\n")
        print(".get_stats() returns:\nself.lisas\nself.zscore, and")
        print("self.sig_mask, a mask of lisas masked for significance at alpha")

    def create_w_matrix(self):
        w = np.empty((self.dist_limit*2-1,self.dist_limit*2-1))

        for i,r in enumerate(range(1-self.dist_limit,self.dist_limit)):
            for j,c in enumerate(range(1-self.dist_limit,self.dist_limit)):

                # double power distance metric: 
                # exponential decrease in weight up to a threshold
                e_dist = np.sqrt(r**2+c**2)
                if e_dist == 0.0:
                    w[i,j] = 0.0
                else:
                    w[i,j] = (1 - (e_dist/self.dthresh)**2)**2

        return w

    def calculate_lisa(self, arr):
        # initialize 
        Z = arr - arr.mean()

        m2 = (Z**2).mean()

        w = self.create_w_matrix()

        idx = np.asarray([i for i in range(1-self.dist_limit,self.dist_limit)])

        lisas = np.empty(arr.shape, dtype=arr.dtype)

        for i in range(self.rmax):
            for j in range(self.cmax):
                ridx = idx + i
                cidx = idx + j

                r_mask = (ridx >= 0) & (ridx < self.rmax)
                c_mask = (cidx >= 0) & (cidx < self.cmax)

                w_i = w[r_mask,:][:,c_mask]
                
                rmin = ridx[r_mask][0]
                rmax = ridx[r_mask][-1]+1
                cmin = cidx[c_mask][0]
                cmax = cidx[c_mask][-1]+1

                s_i = Z[rmin:rmax,cmin:cmax]
                
                lisas[i,j] = (Z[i,j] / m2) * (w_i * s_i).sum()

        return lisas

    def get_stats(self, alpha=0.05):
        two_tailed_p = {0.01:0.99202, 0.05: 0.96012}

        self.lisas = self.calculate_lisa(self.img)

        # parameterize data conditioned on random permutations (and assuming normal dist)
        rand_arr = np.random.permutation(self.img.flatten()).reshape(self.rmax, self.cmax)
        rand_lisas = self.calculate_lisa(rand_arr)

        # generate stats
        self.zscore = (self.lisas - np.mean(rand_lisas))/rand_lisas.std()

        self.sig_mask = np.ma.masked_array(self.lisas, mask=(self.zscore < two_tailed_p[alpha]) & (self.zscore > -two_tailed_p[alpha]))

