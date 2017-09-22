from __future__ import print_function
import cv2
import numpy as np
from nycene_fiducials import lidar_fiducials, image_fiducials

lidar_fid = lidar_fiducials("day_ref") # Nx3
img_fid = image_fiducials("day_ref") # Nx2
nrow = 4096
ncol = 2160
img = np.zeros((nrow, ncol), dtype=np.uint8)

print(" ... Obtaining camera intrinsic parameters")
# CameraMatrix2D expects vector of vectors of points of type Point3f
cam_matrix = cv2.initCameraMatrix2D(objectPoints=[lidar_fid.astype(np.float32)], 
                                    imagePoints=[img_fid.astype(np.float32)], 
                                    imageSize=img.shape, 
                                    aspectRatio=0)

print(" ... Obtaining distortion, rotation and translation matrix")
# Get distortion, rotation and translation matrix
# by using above cam_matrix as initial value
_, _, distortion, rvec, tvec = cv2.calibrateCamera(objectPoints=[lidar_fid.astype(np.float32)], 
                                                   imagePoints=[img_fid.astype(np.float32)], 
                                                   imageSize=img.shape, 
                                                   cameraMatrix=cam_matrix, 
                                                   distCoeffs=None, 
                                                   flags=cv2.CALIB_USE_INTRINSIC_GUESS)

print(" ... Obtaining 2d projected point")
# To verify, plugin the rvec and tvec
# for projecting same lidar fiducials. 
# Expecting the output to be similar to image fiducials
proj2d, jacob = cv2.projectPoints(objectPoints=lidar_fid.astype(np.float), 
                                  rvec=rvec[0], tvec=tvec[0], 
                                  cameraMatrix=cam_matrix.astype(np.float), 
                                  distCoeffs=distortion.astype(np.float), 
                                  aspectRatio=0)

print()
print("Points picked by hand: ")
print(img_fid)
print()
print("Points obtained by projected: ")
print(proj2d)
