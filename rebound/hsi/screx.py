

import numpy as np
import os

def pix_corr(img, trav):
    '''
    pix_corr takes an input of the hyperspectral image and the direction in which we want
    to find the correlation
    
    Args:
        img    :   cube.data
        trav   :   {'right', 'left', 'up', 'down'}
    
    '''
    
    # Code block for converting 3D hyperspectral image to 2D array of correlations
    # Algorithm-1:
        # using for loop to traverse across the pixels and calculating the correlation
        # coefficient using scipy.stats
        # Trying this in test scripts
    # Algorithm-2:
        # Explore np.mean(keepdims=) approach to convert image to 2D array of correlations 
        # Then Use np.ones(img.shape,dtype=bool) to mask the highly correlated pixels
        # Repeat this in all 4 directions
