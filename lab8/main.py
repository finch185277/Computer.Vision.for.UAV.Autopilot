import cv2
import numpy as np
import os
import glob
import imutils


testboard = (5,7)

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

    #cv2.imshow('img',im)
    #cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
f = cv2.FileStorage('Calibration.txt',cv2.FILE_STORAGE_WRITE)
f.write("intrinsic",mtx)
f.write("distortion",dist)

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

cam = cv2.VideoCapture(1)
while True:
    ret,frame = cam.read()
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    if not ret:
        break
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame,winStride=(4,4),padding = (8,8),scale=1.03,useMeanshiftGrouping=False)
    for (x,y,w,h) in rects:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)



    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame,scaleFactor=1.2)



    for (x,y,w,h) in rects:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)

    cv2.imshow("Image",frame)
    cv2.waitKey(1)

f.release()
