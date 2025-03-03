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

horse_h = 20
horse_distance = -1
is_move_forward1 = 0
is_move_forward2 = 0
is_move_left = 0
is_move_right = 0

while True:
    if is_land:
        break

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

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
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

    is_horse = 0
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
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

            # horse
            if(classIDs[i] == 17):
                is_horse = 1
                horse_distance = distance_to_camera(horse_h, focal_length, h);
                cv2.putText(frame, '%f' % horse_distance, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    if(is_horse == 0):
        horse_distance = -1

    if(is_move_left == 0):
        if(horse_distance < 70 and horse_distance > 0):
            drone.move_left(0.9)
            is_move_left = 1
            time.sleep(5)
        elif(horse_distance > 70):
            drone.move_forward(0.2)

    while(is_move_left == 1 and is_move_forward1 == 0):
        # drone.move_forward(0.9)
        frame = drone.read()

        time.sleep(0.01)

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

                if aruco_id == 1 or aruco_id == 4:
                    # forward or backward
                    if tvec[0][0][2] > 50:
                        drone.move_forward(0.2)
                    else:
                        is_move_forward1 = 1

        # show frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", frame)

        # keyboard control
        key = cv2.waitKey(1)
        if key != -1:
            drone.keyboard(key)


    if(is_move_forward1 == 1 and is_move_right == 0):
        time.sleep(5)
        drone.move_right(0.9)
        is_move_right = 1

    if(is_move_right == 1 and is_move_forward2 == 0):
        time.sleep(5)
        drone.move_forward(0.9)
        is_move_forward2 = 1

    if(is_move_forward2 == 1):
        time.sleep(5)
        is_land = 1
        drone.land()

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
