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
drone_distance0 = 50 # for id == 1
drone_distance1 = 95 # for id == 11
drone_distance2 = 60 # for id == 4
distance_error = 10
horizon_error = 5
lr_bound = 20
rotate_bound = 10

# meters
# for id == 1
move_forward_distance = 0.25
move_backward_distance = 0.2
# for id == 4, 11
forward_distance = 0.25
backward_distance = 0.2
lr_distance = 0.2

is_land = 0
is_rotate = 0
rotate_done = 0

while True:
    if is_land:
        break

    frame = drone.read()
    markerCorners, markerlds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters = parameters)
    if markerlds is not None:

        aruco_id = 100
        for id in markerlds[0]:
            if id < aruco_id:
                aruco_id = id

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
                    degree = 180 - degree
                    if degree > rotate_bound:
                        drone.rotate_ccw(degree)
                else:
                    degree = 180 + degree
                    if degree > rotate_bound:
                        drone.rotate_cw(degree)

                # forward or backward
                if tvec[0][0][2] > drone_distance0:
                    drone.move_forward(move_forward_distance)
                else:
                    drone.move_backward(move_backward_distance)

            elif aruco_id == 4:
                # right or left
                if tvec[0][0][0] > lr_bound or tvec[0][0][0] < -lr_bound:
                    if tvec[0][0][0] > 0:
                        drone.move_right(lr_distance)
                    else:
                        drone.move_left(lr_distance)

                # forward or backward
                lowest_distance = drone_distance2 - distance_error
                uppest_distance = drone_distance2 + distance_error
                if tvec[0][0][2] >= lowest_distance and tvec[0][0][2] <= uppest_distance:
                    if tvec[0][0][0] < -(horizon_error) or tvec[0][0][0] > horizon_error:
                        if tvec[0][0][0] > 0:
                            drone.move_right(tvec[0][0][0] / 100)
                        else:
                            drone.move_left(-(tvec[0][0][0]) / 100)
                        time.sleep(0.1)
                        drone.land()
                        is_land = 1
                elif tvec[0][0][2] > drone_distance2:
                    drone.move_forward(forward_distance)
                elif tvec[0][0][2] < drone_distance2:
                    drone.move_backward(backward_distance)

            elif aruco_id == 11:
                if rotate_done == 1:
                    continue

                if is_rotate == 1:
                    drone.rotate_cw(90)
                    rotate_done = 1
                    continue

                # right or left
                if tvec[0][0][0] > lr_bound or tvec[0][0][0] < -lr_bound:
                    if tvec[0][0][0] > 0:
                        drone.move_right(lr_distance)
                    else:
                        drone.move_left(lr_distance)

                # forward or backward
                lowest_distance = drone_distance1 - distance_error
                uppest_distance = drone_distance1 + distance_error
                if tvec[0][0][2] >= lowest_distance and tvec[0][0][2] <= uppest_distance:
                    if tvec[0][0][0] < -(horizon_error) or tvec[0][0][0] > horizon_error:
                        if tvec[0][0][0] > 0:
                            drone.move_right(tvec[0][0][0] / 100)
                        else:
                            drone.move_left(-(tvec[0][0][0]) / 100)
                        time.sleep(1)
                        is_rotate = 1
                elif tvec[0][0][2] > drone_distance1:
                    drone.move_forward(forward_distance)
                elif tvec[0][0][2] < drone_distance1:
                    drone.move_backward(backward_distance)


    # show frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", frame)

    # keyboard control
    key = cv2.waitKey(1)
    if key != -1:
        drone.keyboard(key)

    # sleep
    time.sleep(0.01)

f.release()
