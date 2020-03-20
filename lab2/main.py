import cv2
import numpy as np

def rgb_to_grey(img):
    height, width, depth = img.shape
    ret = np.zeros((height, width, 1), np.uint8)
    for x in range(width):
        for y in range(height):
            grey_value = 0;
            for z in range(depth):
                grey_value += img[y, x, z]
            ret[y, x, 0] = grey_value / 3
    return ret

def histogram_equalization(img):
    img = rgb_to_grey(img)
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    height, width, depth = ret.shape

    pixels = [0] * 256
    acc_sum_p = [0] * 256

    for x in range(width):
        for y in range(height):
            pixels[img[y, x, 0]] += 1

    current_pixels = 0
    total_pixels = height * width
    for i in range(256):
        current_pixels += pixels[i]
        acc_sum_p[i] = float(current_pixels)/float(total_pixels)

    for x in range(width):
        for y in range(height):
            ret[y, x, 0] = int(acc_sum_p[img[y, x, 0]] * 255)

    return ret

def sobel_edge_detection(img, type):
    img = histogram_equalization(img)
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    height, width, depth = ret.shape

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            px = 0
            px += -1 * img[y-1, x-1, 0]
            px +=  1 * img[y-1, x+1, 0]
            px += -2 * img[ y , x-1, 0]
            px +=  2 * img[ y , x+1, 0]
            px += -1 * img[y+1, x-1, 0]
            px +=  1 * img[y+1, x+1, 0]

            py = 0
            py += -1 * img[y-1, x-1, 0]
            py += -2 * img[y-1,  x , 0]
            py += -1 * img[y-1, x+1, 0]
            py +=  1 * img[y+1, x-1, 0]
            py +=  2 * img[y+1,  x , 0]
            py +=  1 * img[y+1, x+1, 0]

            outcome = 0
            if type == 0:
                outcome = max(px, py)
            if type == 1:
                outcome = px
            if type == 2:
                outcome = py

            threshold = 127
            if outcome > threshold:
                ret[y, x, 0] = 255
            else:
                ret[y, x, 0] = 0

    return ret

img1 = cv2.imread('mj.tif')
img2 = cv2.imread('HyunBin.jpg')

demo1 = histogram_equalization(img1)
demo2 = sobel_edge_detection(img2, 0)
demox = sobel_edge_detection(img2, 1)
demoy = sobel_edge_detection(img2, 2)

cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

cv2.imshow('Demo 1', demo1)
cv2.imshow('Demo 2', demo2)
cv2.imshow('Demo x', demox)
cv2.imshow('Demo y', demoy)

cv2.waitKey(0)
cv2.destroyAllWindows()
