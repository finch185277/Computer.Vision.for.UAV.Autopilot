import cv2
import numpy as np
import os
import glob
import tello
from tello_control_ui import TelloUI
import time
import math

#define and store value of the board dimensions inside the testboard list.
testboard = (5,7)

#criteria is the type termcriteria
criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 50, 0.001)

#This vector store vectors of 3D points for each image
objpoints = []

#This vector store vectors of 2D points for each image
imgpoints = []


#define world coordinates for 3D points
objp =  np.zeros((1, testboard[0] * testboard[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:testboard[0], 0:testboard[1]].T.reshape(-1, 2);
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
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11),(-1, -1),criteria)
        imgpoints.append(corners2)
        im = cv2.drawChessboardCorners(im, testboard, corners2, ret)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

f = cv2.FileStorage('calibration.txt', cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", mtx)
f.write("distortion", dist)

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

drone = tello.Tello('', 8889)

time.sleep(5)

# centimeters
drone_distance = 50
distance_error = 10

# meters
move_distance = 0.2
forward_distance = 0.2
backward_distance = 0.05

is_land = 0

while True:
    if is_land:
        break

    frame = drone.read()
    markerCorners, markerlds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters = parameters)
    if markerlds is not None:
        aruco_id = markerlds[0][0]

        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerlds)
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 14.5, mtx, dist)
        if rvec is not None:
            cv2.putText(frame, 'X: %f' % (tvec[0][0][0]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Y: %f' % (tvec[0][0][1]), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Z: %f' % (tvec[0][0][2]), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

            if aruco_id == 1:
                # calculate degree
                rmat = cv2.Rodrigues(rvec[0])
                v = [rmat[0][0][2], rmat[0][1][2], rmat[0][2][2]]
                rad = math.atan2(v[0], v[2])
                degree = math.degrees(rad)

                # rotate
                if degree > 0:
                    drone.rotate_ccw(180 - degree)
                else:
                    drone.rotate_cw(180 + degree)

                # forward or backward
                if tvec[0][0][2] > drone_distance:
                    drone.move_forward(move_distance)
                else:
                    drone.move_backward(move_distance)

            if aruco_id == 4:
                # right or left
                if tvec[0][0][0] > 0:
                    drone.move_right(tvec[0][0][0] / 100)
                else:
                    drone.move_left(-(tvec[0][0][0]) / 100)

                # forward or backward
                lowest_distance = drone_distance - distance_error
                uppest_distance = drone_distance + distance_error
                if tvec[0][0][2] >= lowest_distance and tvec[0][0][2] <= uppest_distance:
                    print("land!!")
                    drone.land()
                    is_land = 1
                elif tvec[0][0][2] > drone_distance:
                    print("distance: ", tvec[0][0][2])
                    drone.move_forward(forward_distance)
                elif tvec[0][0][2] < drone_distance:
                    print("distance: ", tvec[0][0][2])
                    drone.move_backward(back_distance)

    # show frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", frame)

    # keyboard control
    key = cv2.waitKey(1)
    if key != -1:
        drone.keyboard(key)

f.release()
