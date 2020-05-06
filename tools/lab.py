import tello
import cv2
from tello_control_ui import TelloUI
import time

def main():
	drone = tello.Tello('', 8889)

	time.sleep(5)

	while(1):
		frame = drone.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		cv2.imshow("frame",frame)
		key = cv2.waitKey(1)

		if key != -1:
		    drone.keyboard(key)

if __name__ == "__main__":
    main()
