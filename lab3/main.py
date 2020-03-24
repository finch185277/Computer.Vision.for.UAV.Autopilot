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

    current_pixels = 0
    total_pixels = height * width
    for i in range(256):
        current_pixels += pixel_count[i]
        pixel_pros[i] = float(current_pixels)/float(total_pixels)

    threshold = 0
    max_delta = 0
    for i in range(256):
        w0, w1, t0, t1, u0, u1, u, cur_delta = [0.0 for _ in range(8)]
        for j in range(256):
            if j <= i:
                w0 += pixel_pros[j]
                t0 += j * pixel_pros[j]
            else:
                w1 += pixel_pros[j]
                t1 += j * pixel_pros[j]

        try:
            u0 = t0 / w0
        except ZeroDivisionError:
            u0 = 0

        try:
            u1 = t1 / w1
        except ZeroDivisionError:
            u1 = 0

        u = t0 + t1
        cur_delta = w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2)
        if cur_delta > max_delta:
            max_delta = cur_delta
            threshold = i

    for x in range(width):
        for y in range(height):
            if img[y, x, 0] > threshold:
                ret[y, x, 0] = 255
            else:
                ret[y, x, 0] = 0

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
