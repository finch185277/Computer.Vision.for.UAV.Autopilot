import cv2
import numpy as np
import os
import glob
import imutils

def distance_to_camera(knownH, focal_length, perH):
	# compute and return the distance from the maker to the camera
	return (knownH * focal_length) / perH

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

focal_length = (mtx[0, 0] + mtx[1, 1]) / 2

cam = cv2.VideoCapture(1)
while True:
    ret,frame = cam.read()
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    if not ret:
        break

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame, winStride=(4,4), padding = (8,8), scale=1.03, useMeanshiftGrouping=False)

    face_H = 15
    rect_W = 80
    faces_perH = 1
    rects_perW = 1

    # faces
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)
        faces_perH = h

    # rects
    mx = 0
    my = 0
    mw = 0
    mh = 0
    for (x,y,w,h) in rects:
        if h > mh:
            mx = x
            my = y
            mw = w
            mh = h
    frame = cv2.rectangle(frame,(mx,my),(mx+mw,my+mh), (0,255,255), 2)
    rects_perW = mw

    # faces
    cv2.putText(frame, '%f' % distance_to_camera(face_H, focal_length, faces_perH), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    # rects
    cv2.putText(frame, '%f' % distance_to_camera(rect_W, focal_length, rects_perW), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Image",frame)
    cv2.waitKey(1)

f.release()
