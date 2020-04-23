import cv2
import numpy as np
import os
import glob
import tello
from tello_control_ui import TelloUI
import time

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
  im = cv2.imread(fname)
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, testboard)

  if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
    im = cv2.drawChessboardCorners(im,testboard,corners2, ret)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

f = cv2.FileStorage('calibration.txt',cv2.FILE_STORAGE_WRITE)
f.write("intrinsic",mtx)
f.write("distortion",dist)

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

drone = tello.Tello('', 8889)

time.sleep(5)

while True:
  frame = drone.read()
  markerCorners, markerlds, rejectedCandidates = cv2.aruco.detectMarkers(frame,dictionary,parameters = parameters)

  frame = cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerlds)

  rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners,13.8,mtx, dist)
  if rvec is not None:
    frame = cv2.aruco.drawAxis(frame,mtx,dist,rvec,tvec,6)
    cv2.putText(frame,'X: %f' % (tvec[0][0][0]),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,'Y: %f' % (tvec[0][0][1]),(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,'Z: %f' % (tvec[0][0][2]),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1,cv2.LINE_AA)

  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

  cv2.imshow("Image",frame)
  cv2.waitKey(1)

f.release()
