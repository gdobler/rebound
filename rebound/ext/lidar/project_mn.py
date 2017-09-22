import cv2
import numpy as np
from nycene_fiducials import lidar_fiducials, image_fiducials

lidar_fid = lidar_fiducials("day_ref") # Nx3
img_fid = image_fiducials("day_ref") # Nx2
# taken from project.py. How was it obtained? TBFO
focal_len = 1.63556904e+04
nrow = 4096
ncol = 2160
# camera intrinsic parameters
cam_mat = np.array([[focal_len, 0, ncol/2], 
                    [0, focal_len, nrow/2], 
                    [0, 0, 1]], dtype=np.float)

# Obtaining rotation and translation matrix
# i.e finding object pose from 3D-2D point correspondences
# !!! Cannot give rotvec as 3x3, it only accepts 1x3 or 3x1 !!!
_, rot_vec, trans_vec = cv2.solvePnP(objectPoints=lidar_fid.astype(np.float),
                                     imagePoints=img_fid.astype(np.float),
                                     cameraMatrix=cam_mat,
                                     distCoeffs=np.zeros((1,4), dtype=np.float),
                                     rvec=np.zeros((1,3)), tvec=np.zeros((1,3)), 
                                     useExtrinsicGuess=True)

# To verify, plugin the rot_vec and trans_vec
# for projecting same lidar fiducials. 
# Expecting the output to be similar to image fiducials
proj2d, jacob = cv2.projectPoints(objectPoints=lidar_fid.astype(np.float), 
                                  rvec=rot_vec, tvec=trans_vec, 
                                  cameraMatrix=cam_mat,
                                  distCoeffs=np.zeros((1,4), dtype=np.float))
