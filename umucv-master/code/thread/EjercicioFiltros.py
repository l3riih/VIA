#!/usr/bin/env python

import numpy as np
import cv2 as cv
from umucv.util import ROI, putText
from umucv.stream import autoStream

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)
region = ROI("input")
def election(key, chose):
    key_mappings = {
        ord('1'): 1,
        ord('2'): 2,
        ord('3'): 3,
        ord('4'): 4,
        ord('5'): 5,
        ord('6'): 6
    }
    return key_mappings.get(key, chose)
# Función para aplicar los filtros
def apply_filter(chose, frame):
    if chose == 0:
        return frame
    elif chose == 1:
        k = cv.getTrackbarPos('k', 'input')
        return cv.boxFilter(frame, -1, (k, k))
    elif chose == 2:
        k = cv.getTrackbarPos('k', 'input')
        sigma = cv.getTrackbarPos('sigma', 'input')
        return cv.GaussianBlur(frame, (k, k), sigma)
    elif chose == 3:
        k = cv.getTrackbarPos('k', 'input')
        return cv.medianBlur(frame, k)
    elif chose == 4:
        k = cv.getTrackbarPos('k', 'input')
        sigma_color = cv.getTrackbarPos('sigma_color', 'input')
        sigma_space = cv.getTrackbarPos('sigma_space', 'input')
        return cv.bilateralFilter(frame, k, sigma_color, sigma_space)
    elif chose == 5:
        k = cv.getTrackbarPos('k', 'input')
        return cv.erode(frame, np.ones((k, k), np.uint8))
    elif chose == 6:
        k = cv.getTrackbarPos('k', 'input')
        return cv.dilate(frame, np.ones((k, k), np.uint8))

# Trackbars
cv.createTrackbar('k', 'input', 1, 50, lambda x: None)
cv.createTrackbar('sigma', 'input', 1, 100, lambda x: None)
cv.createTrackbar('sigma_color', 'input', 1, 100, lambda x: None)
cv.createTrackbar('sigma_space', 'input', 1, 100, lambda x: None)

chose = 0
for key, frame in autoStream():
    chose = election(key, chose)
    if region.roi:
        [x1, y1, x2, y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2 + 1, x1:x2 + 1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []

        roi_filtered = apply_filter(chose, frame[y1:y2 + 1, x1:x2 + 1])
        frame[y1:y2 + 1, x1:x2 + 1] = roi_filtered

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))

    h, w, _ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input', frame)



cv.destroyAllWindows()
