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

def nearest_neighbor_interpolation(img, r):
    ret = np.zeros((int(img.shape[0] * r), int(img.shape[1] * r), img.shape[2]), np.uint8)
    height, width, depth = ret.shape
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                ret[y, x, z] = img[int(y/r), int(x/r), z]
    return ret

def bilinear_interpolation(img, r):
    ret = np.zeros((int(img.shape[0] * r), int(img.shape[1] * r), img.shape[2]), np.uint8)
    height, width, depth = ret.shape
    for x in range(width):
        for y in range(height):
            if x == (x/r) * r and y == (y/r) * r:
                for z in range(depth):
                    ret[y, x, z] = img[int(y/r), int(x/r), z]
            else:
                for z in range(depth):
                    ret[y, x, z] += img[int(y/r), int(x/r), z] * (int(x/r) + 1 - float(x/r)) * (int(y/r) + 1 - float(y/r))
                    ret[y, x, z] += img[int(y/r + 1), int(x/r), z] * (int(x/r) + 1 - float(x/r)) * (float(y/r) - int(y/r))
                    ret[y, x, z] += img[int(y/r), int(x/r + 1), z] * (float(x/r) - int(x/r)) * (int(y/r) + 1 - float(y/r))
                    ret[y, x, z] += img[int(y/r + 1), int(x/r + 1), z] * (float(x/r) - int(x/r)) * (float(y/r) - int(y/r))

    return ret

img1 = cv2.imread('kobe.jpg')
img2 = cv2.imread('IU.png')

demo1 = rgb_to_grey(img1)
demo2 = nearest_neighbor_interpolation(img2, 3)
demo3 = bilinear_interpolation(img2, 0.7)

cv2.imshow('My Image 1', img1)
cv2.imshow('My Image 2', img2)

cv2.imshow('My Demo 1', demo1)
cv2.imshow('My Demo 2', demo2)
cv2.imshow('My Demo 3', demo3)

cv2.waitKey(0)
cv2.destroyAllWindows()
