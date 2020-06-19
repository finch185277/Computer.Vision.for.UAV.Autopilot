import cv2
import numpy as np
import os
import glob
import tello
from tello_control_ui import TelloUI
import time
import math
import argparse
import imutils

def distance_to_camera(knownH, focal_length, perH):
    # compute and return the distance from the maker to the camera
    return (knownH * focal_length) / perH

def find_object(object_id): # return distance
    object_distance = -1

    frame = drone.read()

    (W, H) = (None, None)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # probability is greater than the minimum probability
            # filter out weak predictions by ensuring the detected
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if(classIDs[i] == object_id):
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # calulate distance
                object_distance = distance_to_camera(horse_h, focal_length, h);
                cv2.putText(frame, '%f' % object_distance, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    # show frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", frame)

    # keyboard control
    key = cv2.waitKey(1)
    if key != -1:
        drone.keyboard(key)

    time.sleep(0.01)

    return object_distance

def find_aruco(id, bound): # return status: -1, 0, 1
    is_id_find = 0
    is_job_done = -1
    cnt = 0
    idx = 0

    frame = drone.read()
    markerCorners, markerlds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters = parameters)
    if markerlds is not None:
        for i in markerlds:
            for j in i:
                if j == id:
                    is_id_find = 1
                    idx = cnt
            cnt += 1

        if(is_id_find == 0): # aruco not found
            # show frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", frame)

            # keyboard control
            key = cv2.waitKey(1)
            if key != -1:
                drone.keyboard(key)

            return is_job_done

        is_job_done = 0

        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerlds)
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 14.5, mtx, dist)
        if rvec is not None:
            cv2.putText(frame, 'X: %f' % (tvec[idx][0][0]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Y: %f' % (tvec[idx][0][1]), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Z: %f' % (tvec[idx][0][2]), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

            # right or left
            if tvec[idx][0][0] > lr_bound or tvec[idx][0][0] < -lr_bound:
                if tvec[idx][0][0] > 0:
                    if tvec[idx][0][0] / 100 > lr_distance:
                        drone.move_right(tvec[idx][0][0]/100)
                    else:
                        drone.move_right(lr_distance)
                else:
                    if -tvec[idx][0][0]/100 > lr_distance:
                        drone.move_left(-tvec[idx][0][0]/100)
                    else:
                        drone.move_left(lr_distance)
                time.sleep(aruco_wait_time)
            else:
                # forward or continue
                if tvec[idx][0][2] > bound:
                    drone.move_forward(forward_distance)
                else:
                    is_job_done = 1

    # show frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", frame)

    # keyboard control
    key = cv2.waitKey(1)
    if key != -1:
        drone.keyboard(key)

    time.sleep(0.01)

    return is_job_done


ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

focal_length = (mtx[0, 0] + mtx[1, 1]) / 2

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

drone = tello.Tello('', 8889)

time.sleep(3)

state_wait_time = 3
aruco_wait_time = 1

# object id
horse_id = 17
traffic_light_id = 9

# centimeters
lr_bound = 15
horse_h = 20
horse_distance = 60
lane_distance = 130
aruco_distance = 60

# meters
lr_distance = 0.2
forward_distance = 0.25
horse_lr_distance = 0.9
horse_forward_distance = 1.2
fast_forward_distance = 1.1

# bool
is_land = 0

# horse part
is_move_forward1 = 0
is_move_forward2 = 0
is_move_left1 = 0
is_move_right1 = 0

# lane part
is_move_forward3 = 0
is_move_forward4 = 0
is_move_forward5 = 0
is_move_forward6 = 0
is_move_forward7 = 0
is_move_left2 = 0
is_rotate_cw = 0
is_rotate_ccw = 0
is_rotate_180 = 0
is_flip = 0

while True:
    if is_land:
        break

    frame = drone.read()

    # horse part
    while(is_move_right1 == 0):
        object_distance = find_object(horse_id)
        if(object_distance == -1):
            continue
        if(object_distance > horse_distance):
            drone.move_forward(forward_distance)
        else:
            time.sleep(state_wait_time)
            drone.move_right(horse_lr_distance)
            is_move_right1 = 1
            time.sleep(state_wait_time)

    while(is_move_right1 == 1 and is_move_forward1 == 0):
        is_job_done = find_aruco(4, aruco_distance)
        if(is_job_done == 1):
            is_move_forward1 = 1
            time.sleep(state_wait_time)

    if(is_move_forward1 == 1 and is_move_left1 == 0):
        drone.move_left(horse_lr_distance)
        is_move_left1 = 1
        time.sleep(state_wait_time)

    if(is_move_left1 == 1 and is_move_forward2 == 0):
        is_job_done = find_aruco(11, lane_distance)
        time.sleep(state_wait_time)
        drone.move_forward(horse_forward_distance)
        is_move_forward2 = 1
        time.sleep(state_wait_time)

    # lane part
    while(is_move_forward2 == 1 and is_move_forward3 == 0):
        is_job_done = find_aruco(11, lane_distance)
        if(is_job_done == 1):
            is_move_forward3 = 1
            time.sleep(state_wait_time)

    if (is_move_forward3 == 1 and is_rotate_180 == 0):
        drone.rotate_cw(180)
        is_rotate_180 = 1
        time.sleep(state_wait_time)

    while(is_rotate_180 == 1 and is_flip == 0):
        object_distance = find_object(traffic_light_id)
        if(object_distance == -1):
            drone.move_left(lr_distance)
            continue
        time.sleep(state_wait_time)
        drone.flip('r')
        is_flip = 1
        time.sleep(state_wait_time)

    # last part
    while(is_flip == 1 and is_move_left2 == 0):
        is_job_done = find_aruco(14, aruco_distance)
        if(is_job_done == -1): # aruco not found
            drone.move_left(lr_distance)
            continue
        is_move_left2 = 1 # aruco already found
        time.sleep(state_wait_time)

    while(is_move_left2 == 1 and is_move_forward4 == 0):
        is_job_done = find_aruco(14, aruco_distance)
        if(is_job_done == 1):
            is_move_forward4 = 1
            time.sleep(state_wait_time)

    if(is_move_forward4 == 1 and is_rotate_ccw == 0):
        drone.rotate_ccw(90)
        is_rotate_ccw = 1
        time.sleep(state_wait_time)

    if(is_rotate_ccw == 1 and is_move_forward5 == 0):
        is_job_done = find_aruco(45, aruco_distance)
        if(is_job_done == 1):
            is_move_forward5 = 1
            time.sleep(state_wait_time)

    if(is_move_forward5 == 1 and is_rotate_cw == 0):
        drone.rotate_cw(90)
        is_rotate_cw = 1
        time.sleep(state_wait_time)

    if(is_rotate_cw == 1 and is_move_forward6 == 0):
        drone.move_forward(fast_forward_distance)
        is_move_forward6 = 1
        time.sleep(state_wait_time)

    if(is_move_forward6 == 1 and is_move_forward7 == 0):
        is_job_done = find_aruco(22, aruco_distance)
        if(is_job_done == 1):
            is_move_forward7 = 1
            time.sleep(state_wait_time)

    # done!!
    if(is_move_forward7 == 1):
        is_land = 1
        drone.land()
        time.sleep(state_wait_time)

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
