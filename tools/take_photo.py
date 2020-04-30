import cv2

cap = cv2.VideoCapture(1)  # device
idx = 0
while(True):
    ret, frame = cap.read()
    # ret is True if read() successed
    cv2.imshow('frame', frame)
    key = cv2.waitKey(33)
    if key == 32:  # space
        imgname = "img" + str(idx) + ".jpg"
        cv2.imwrite(imgname, frame)
        idx += 1
    if key == 27:  # esc
        cv2.destroyAllWindows()
        break
