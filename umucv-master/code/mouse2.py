#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText
from collections import deque
xi, yi = 0, 0
d = deque(maxlen=2)
def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global xi, yi
        xi, yi = x, y
        d.append((x, y))
        print(x, y)


cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)

for key, frame in autoStream():

    for p in d:
        cv.circle(frame, p, 10, (0, 0, 255), -1)
        putText(frame, (str(p[0]) + " " + str(p[1])), p, (255, 0, 0))

    cv.imshow('webcam', frame)
cv.destroyAllWindows()
