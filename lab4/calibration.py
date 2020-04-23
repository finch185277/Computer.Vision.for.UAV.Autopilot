import cv2
import numpy as np
import os
import glob

#define and store value of the board dimensions inside the testboard list.
testboard = (5,7)

#criteria is the type termcriteria

#Termcriteria(int type,int maxCount, double epsilon)
#This criteria is used to define when to stop when refining checkboard corners, that is,
#get the coordinates fit with the corners more precisely.

#type defines the condition to stop, cv2.TERM_CRITERIA_EPS means to stop when the epsilon is satisfied,
#while cv2.TERM_CRITERIA_MAX_ITER means to stop the refining procedure after max times of iterations
# use these together suing a + notation

# 50 means the max times of iterations is 50
# 0.001 stands for epsilon
criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 50, 0.001)

#This vector store vectors of 3D points for each image
objpoints = []

#This vector store vectors of 2D points for each image
imgpoints = []


#define world coordinates for 3D points
objp =  np.zeros((1, testboard[0] * testboard[1], 3),np.float32)
objp[0,:,:2] = np.mgrid[0:testboard[0],0:testboard[1]].T.reshape(-1,2);
prev_image_type = None

#get images
images = glob.glob('./images/*.jpg')

gray = cv2.imread('./images/1.jpg')

for fname in images:
  print(fname)
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, testboard)

  if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
    img = cv2.drawChessboardCorners(img,testboard,corners2, ret)

  cv2.imshow('img',img)
  cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Mtx :\n")
print(mtx)
print("Dist: \n")
print(dist)
print("rvecs: \n")
print(rvecs)
print("tvecs: \n");
print(tvecs)
f = cv2.FileStorage('calibration.txt',cv2.FILE_STORAGE_WRITE)
f.write("intrinsic",mtx)
f.write("distortion",dist)
f.release()
