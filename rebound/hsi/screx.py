
import numpy as np
import os
import utils
import scipy.ndimage.measurements as mm

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
    img = 1.0 * cube.data

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



def sptr_mean(path, fname, boolean_mask):
	'''
	sptr_mean takes an input of the hyperspectral image and the boolean array
	from hyper_pixcorr function to give an output image with mean spectral intensities
	across the sources in each spectral channel

	Input Parameters:
	------------
	path = str
		Path to the directory where hyperspectral images are located

	fname = str
		File name of raw hyperspectral image

	boolean_mask = np.array
		Output boolean mask of sources from hyper_pixcorr function

	Output:
	------------
	src_sptr_mean = np.array
		Mean Spectrum of sources across corresponsing source pixels
	'''

	# Reading the Raw hyperspectral image
	cube = utils.read_hyper(path, fname)

	# Storing the hyperspectral image as a memmap for future computations
	img = 1.0 * cube.data

	#Labeling the sources in Boolean Mask
	mask = boolean_mask
	labels, count = mm.label(mask)

	index = np.arange(count+1)
	sptr_stack = []

	for i in range(0,img.shape[0]):
    	channel = img[i,:,:]
    	src_mean = mm.mean(channel, labels, index)
    	sptr_stack.append(src_mean)
    
	return np.array(sptr_stack)



