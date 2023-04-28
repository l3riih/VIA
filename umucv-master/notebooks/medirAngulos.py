#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText

points = deque(maxlen=2)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

# Distancia focal en píxeles
f = 481

# Dimensiones de la imagen
W = 640
H = 480

for key, frame in autoStream():
    for p in points:
        cv.circle(frame, p, 3, (0, 0, 255), -1)
    if len(points) == 2:
        cv.line(frame, points[0], points[1], (0, 0, 255))
        c = np.mean(points, axis=0).astype(int)

        p1, p2 = points

        X1 = p1[0] - W/2
        Y1 = p1[1] - H/2
        Z1 = f

        X2 = p2[0] - W/2
        Y2 = p2[1] - H/2
        Z2 = f

        v1 = np.array([X1, Y1, Z1])
        v2 = np.array([X2, Y2, Z2])

        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)

        cos_theta = dot_product / (mag_v1 * mag_v2)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        putText(frame, f'{angle_deg:.1f} grados', c)

    cv.imshow('webcam', frame)

cv.destroyAllWindows()
