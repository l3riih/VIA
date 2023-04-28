#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    h, w, _ = frame.shape
    cv.imshow('input',frame)

cv.destroyAllWindows()

