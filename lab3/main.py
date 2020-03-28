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


def seed_filling_algorithm2(img):
    img = otsu_threshold(img)
    label = np.zeros((img.shape[0], img.shape[1]))
    ret = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    height, width, depth = ret.shape

    cur_label = 1
    equl_set = [set() for _ in xrange(height * width)]

    for x in range(width):
        for y in range(height):
            if img[y, x, 0] == 255:
                if x == 0 and y == 0:
                    label[y, x] = cur_label
                    equl_set[cur_label].add(cur_label)
                    cur_label += 1

                if x > 0 and y == 0:
                    if label[y, x - 1] != 0:
                        label[y, x] = label[y, x - 1]
                    else:
                        label[y, x] = cur_label
                        equl_set[cur_label].add(cur_label)
                        cur_label += 1

                if x == 0 and y > 0:
                    if label[y - 1, x] != 0:
                        label[y, x] = label[y - 1, x]
                    else:
                        label[y, x] = cur_label
                        equl_set[cur_label].add(cur_label)
                        cur_label += 1

                if x > 0 and y > 0:
                    if label[y, x - 1] != 0 and label[y - 1, x] != 0:
                        min_label = min(label[y, x - 1], label[y - 1, x])
                        label[y, x] = min_label

                        if label[y, x - 1] != label[y - 1, x]:
                            xl = 0  # x label
                            yl = 0  # y label

                            if label[y, x - 1] in equl_set[int(label[y, x - 1])]:
                                xl = label[y, x - 1]
                            else:
                                for lab in range(1, int(label[y, x - 1])):
                                    if label[y, x - 1] in equl_set[lab]:
                                        xl = lab
                                        break

                            if label[y - 1, x] in equl_set[int(label[y - 1, x])]:
                                yl = label[y - 1, x]
                            else:
                                for lab in range(1, int(label[y - 1, x])):
                                    if label[y - 1, x] in equl_set[lab]:
                                        yl = lab
                                        break

                            if xl != yl:
                                if xl < yl:
                                    equl_set[int(xl)] = equl_set[int(xl)].union(equl_set[int(yl)])
                                    equl_set[int(yl)].clear()
                                else:
                                    equl_set[int(yl)] = equl_set[int(yl)].union(equl_set[int(xl)])
                                    equl_set[int(xl)].clear()

                    if label[y, x - 1] != 0 and label[y - 1, x] == 0:
                        label[y, x] = label[y, x - 1]

                    if label[y - 1, x] != 0 and label[y, x - 1] == 0:
                        label[y, x] = label[y - 1, x]

                    if label[y, x - 1] == 0 and label[y - 1, x] == 0:
                        label[y, x] = cur_label
                        equl_set[cur_label].add(cur_label)
                        cur_label += 1

    colors = np.zeros((cur_label, 3))
    for i in range(1, cur_label):
        for j in range(3):
            colors[i, j] = randrange(256)

    fixed_label = np.zeros((cur_label))
    for lab in range(1, cur_label):
        if lab in equl_set[lab]:
            fixed_label[lab] = lab
            continue
        else:
            for ex_lab in range(1, cur_label):
                if lab in equl_set[ex_lab]:
                    fixed_label[lab] = ex_lab
                    break

    for x in range(width):
        for y in range(height):
            if img[y, x, 0] == 255:
                for z in range(depth):
                    ret[y, x, z] = colors[int(fixed_label[int(label[y, x])]), z]

    return ret


img1 = cv2.imread('input.jpg')

demo1 = otsu_threshold(img1)
demo2 = seed_filling_algorithm(img1)
demo3 = seed_filling_algorithm2(img1)

cv2.imshow('Image 1', img1)

cv2.imshow('Demo 1', demo1)
cv2.imshow('Demo 2', demo2)
cv2.imshow('Demo 3', demo3)

cv2.waitKey(0)
cv2.destroyAllWindows()
