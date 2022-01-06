import random
import cv2
import numpy as np
from matplotlib import pyplot as plt



if __name__ == "__main__":
    img = cv2.imread('/annotations/10_wire_0.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, src = cv2.threshold(imgray, 127, 255, 0)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB).copy()

    contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(dst, contours, 0, (255,0,0), 2, cv2.LINE_8, hierarchy)
    # for idx in range(len(contours)):
    #     c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     cv2.drawContours(dst, contours, idx, c, 2, cv2.LINE_8, hierarchy)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB).copy()
    epsilon = 0.003*cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    cv2.drawContours(dst2, [approx], 0, (255,0,0), 2, cv2.LINE_8, hierarchy)
    plt.imshow(dst2)

    list(approx) # return
