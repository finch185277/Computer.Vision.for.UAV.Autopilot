import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("Image")  # adjust the "Image" image


img = cv2.imread("warp.jpg")

y = img.shape[0]
x = img.shape[1]

print(x)  # 480 // height
print(y)  # 640 // width

background = np.zeros([480, 640, 3])
background = background.astype('uint8')

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame_y = float(frame.shape[0])
    frame_x = float(frame.shape[1])

    dst_pts = np.array([[193.0, 116.0], [450.0, 53.0], [178.0, 272.0], [440.0, 274.0]], dtype = "float32")
    src_pts = np.array([[0.0, 0.0], [frame_x, 0.0], [0.0, frame_y], [frame_x, frame_y]], dtype = "float32")

    cvtMtx = cv2.getPerspectiveTransform(src_pts, dst_pts)

    processed = cv2.warpPerspective(frame, cvtMtx, (640, 480))
    square = np.array([[193, 116], [450, 53], [440, 274], [178, 272]])

    cv2.fillConvexPoly(img, square, (0, 0, 0))
    for i in range(480):
        for j in range(640):
            if not img[i][j][0] and not img[i][j][1] and not img[i][j][2]:
                img[i][j] = processed[i][j]

    cv2.imshow("Image", img)

    cv2.waitKey(1)
