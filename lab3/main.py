import cv2
import numpy as np
from random import randrange


def rgb_to_grey(img):
    height, width, depth = img.shape
    ret = np.zeros((height, width, 1), np.uint8)
    for x in range(width):
        for y in range(height):
            grey_value = 0
            for z in range(depth):
                grey_value += img[y, x, z]
            ret[y, x, 0] = grey_value / 3
    return ret


def otsu_threshold(img):
    img = rgb_to_grey(img)
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    height, width, depth = ret.shape

    pixel_count = [0] * 256
    pixel_pros = [0] * 256

    for x in range(width):
        for y in range(height):
            pixel_count[img[y, x, 0]] += 1

    total_pixels = height * width
    for i in range(256):
        pixel_pros[i] = float(pixel_count[i])/float(total_pixels)

    threshold = 0
    max_var = 0
    for t in range(256):
        nb = 0
        no = 0
        cb = 0
        co = 0
        mb = 0
        mo = 0
        for i in range(256):
            if i < t:
                nb += pixel_count[i]
                cb += pixel_count[i] * i
            else:
                no += pixel_count[i]
                co += pixel_count[i] * i

        try:
            mb = cb / nb
        except ZeroDivisionError:
            mb = 0

        try:
            mo = co / no
        except ZeroDivisionError:
            mo = 0

        cur_var = nb * no * pow((mb - mo), 2)
        if cur_var > max_var:
            threshold = t
            max_var = cur_var

    for x in range(width):
        for y in range(height):
            if img[y, x, 0] < threshold:
                ret[y, x, 0] = 0
            else:
                ret[y, x, 0] = 255

    return ret


def seed_filling_algorithm(img):
    img = otsu_threshold(img)
    label = np.zeros((img.shape[0], img.shape[1]))
    ret = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    height, width, depth = ret.shape

    # BFS (DFS would throw a maximum recursion depth exceeded error)
    for x in range(width):
        for y in range(height):
            if img[y, x, 0] == 255 and label[y, x] == 0:
                color = [randrange(256), randrange(256), randrange(256)]
                queue = []
                queue.append((x, y))
                while queue:
                    i, j = queue.pop()
                    for z in range(depth):
                        ret[j, i, z] = color[z]
                    label[j, i] = 1

                    if i < width - 1:
                        if img[j, i + 1, 0] == 255 and label[j, i + 1] == 0:
                            queue.append((i + 1, j))
                    if i > 0:
                        if img[j, i - 1, 0] == 255 and label[j, i - 1] == 0:
                            queue.append((i - 1, j))
                    if j < height - 1:
                        if img[j + 1, i, 0] == 255 and label[j + 1, i] == 0:
                            queue.append((i, j + 1))
                    if j > 0:
                        if img[j - 1, i, 0] == 255 and label[j - 1, i] == 0:
                            queue.append((i, j - 1))

    return ret


img1 = cv2.imread('input.jpg')

demo1 = otsu_threshold(img1)
demo2 = seed_filling_algorithm(img1)

cv2.imshow('Image 1', img1)

cv2.imshow('Demo 1', demo1)
cv2.imshow('Demo 2', demo2)

cv2.waitKey(0)
cv2.destroyAllWindows()
