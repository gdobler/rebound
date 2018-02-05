#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class MoranLisa(object):
    '''
    img: numpy 3-d array
    Takes as input image cube: nwavs x nrows x ncols
    Either HSI or BB in 3-channel RGB format.

    dist_limit: int (default 5)
        Threshold for spatial distance 

    alpha : float (default 0.05)
        Statistical test alpha threshold.

    Methods:
    .get_lm_stats() gets local I values for 2-d image file (i.e. mean luminosity)
    for each pixel.

    Output is a masked 2-d array, masked for stat significance.

    .get_spectra_stats() iterates through each channel to get local I 
    or each pixel for each channel.
    
    Output is 3-d cube of masked 2-d arrays for each channel.

    '''

    def __init__(self, img, dist_limit=5, alpha=0.05):
 
        self.img = img
        self.lm = img.mean(0)
        self.dthresh = np.sqrt(2*dist_limit**2)
        self.dist_limit = dist_limit
        self.num_ch = self.img.shape[0]
        self.rmax = self.img.shape[1]
        self.cmax = self.img.shape[2]
        self.alpha = alpha

        # create weight matrix
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

        self.w = w

    def calculate_lisa(self, arr):
        # initialize 
        Z = arr - arr.mean()

        m2 = (Z**2).mean()

        idx = np.asarray([i for i in range(1-self.dist_limit,self.dist_limit)])

        lisas = np.empty(arr.shape, dtype=arr.dtype)

        for i in range(self.rmax):
            for j in range(self.cmax):
                ridx = idx + i
                cidx = idx + j

                r_mask = (ridx >= 0) & (ridx < self.rmax)
                c_mask = (cidx >= 0) & (cidx < self.cmax)

                w_i = self.w[r_mask,:][:,c_mask]
                
                rmin = ridx[r_mask][0]
                rmax = ridx[r_mask][-1]+1
                cmin = cidx[c_mask][0]
                cmax = cidx[c_mask][-1]+1

                s_i = Z[rmin:rmax,cmin:cmax]
                
                lisas[i,j] = (Z[i,j] / m2) * (w_i * s_i).sum()

        return lisas

    def calculate_stats(self, arr):
        two_tailed_p = {0.01:0.99202, 0.05: 0.96012}
        self.p = two_tailed_p[self.alpha]

        # parameterize data conditioned on random permutations (and assuming normal dist)
        rand_arr = np.random.permutation(arr.flatten()).reshape(self.rmax, self.cmax)
        rand_lisas = self.calculate_lisa(rand_arr)

        lisas = self.calculate_lisa(arr)

        # generate stats
        zscore = (lisas - rand_lisas.mean())/rand_lisas.std()

        sig_mask = np.ma.masked_array(lisas, mask=(zscore < self.p) & (zscore > -self.p))

        return sig_mask, zscore

    def get_lm_stats(self):
        output = self.calculate_stats(self.lm)
        self.lm_lisas = output[0]
        self.lm_zscore = output[1]


    def get_spectra_stats(self):
        spectra_lisas = np.empty(self.img.shape)

        count = 0
        for i in range(self.num_ch):
            if i % 100 == 0:
                print('Calculating channel {} of {}'.format(i,self.num_ch))
            output = self.calculate_stats(self.img[i,:,:])[0]
            output.data[output.mask] = 0
            spectra_lisas[i,:,:] = output.data
            
        self.spectra_lisas = spectra_lisas
        self.spectra_lisas_mu = np.mean(self.spectra_lisas, axis = 0)

        np.save(os.path.join(os.environ["REBOUND_WRITE"],'circadian','spectra_lisas2.npy'), self.spectra_lisas)




