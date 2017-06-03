

import numpy as np
import os
import utils

def hyper_pixcorr(path, fname, thresh=0.5):
    '''
    hyper_pixcorr takes an input of the hyperspectral image and the threshold
    correlation values and gives an output boolean array of pixels that are 
    correlated with their neighbors.
    
    Input Parameters:
    ------------

        path = str
           Path to the directory where hyperspectral images are located

        fname = str
           File name of raw hyperspectral image

        thresh = float
           Threshold correlation value for masking the correlated sources

    Output:
    ------------
	final_mask = np.array
	    Boolean array of pixel locations with correlations along both axes

	    Note: The dimension of this array is trimmed by 1 row and 1 column than
	    the input image
    '''
        
    # Reading the Raw hyperspectral image
    cube = utils.read_hyper(path, fname)

    # Storing the hyperspectral image as a memmap for future computations
    img = 1 * cube.data

    # Normalizing the Image
    img -= img.mean(0, keepdims=True)
    img /= img.std(0, keepdims=True)

    # Computing the correlations between the left-right pixels
    corr_x = (img[:,:-1,:] * img[:,1:,:]).mean(0)

    # Computing the correlations between the top-down pixels
    corr_y = (img[:,:,:-1] * img[:,:,1:]).mean(0)

    # Creating a Mask for all the pixels/sources with correlation greater than threshold
    corr_mask_x = corr_x[:,:-1] > thresh
    corr_mask_y = corr_y[:-1,:] > thresh

    # Merging the correlation masks in left-right and top-down directions
    final_mask = corr_mask_x | corr_mask_y

    return final_mask
