#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv
from collections import deque
import time

from umucv.util import ROI, putText
from umucv.stream import autoStream
from umucv.util import Video

d = deque(maxlen=5)
video = Video(fps=15, codec="MJPG",ext="avi")
cv.namedWindow("input")
cv.moveWindow('input', 0, 0)
start_time = 0
duration = 3  # Duración de la grabación en segundos

region = ROI("input")
record = False
move = 0
threshold = 50000
for key, frame in autoStream():


    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []
            continue

        gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gframe = cv.GaussianBlur(gframe, (21, 21), 0)
        d.append(gframe)

        if len(d) == 5:
            meanFrames = np.mean(d, axis=0).astype(np.uint8)
            diff = cv.absdiff(gframe, meanFrames)
            cv.imshow('diff', diff[y1:y2 + 1, x1:x2 + 1])
            move = np.sum(diff[y1:y2 + 1, x1:x2 + 1])

        if move > threshold:
            if not record:
                record = True
                start_time = time.time()

        if record:
            # Si se está grabando, mostrar el tiempo restante
            elapsed_time = time.time() - start_time
            remaining_time = max(0, duration - elapsed_time)
            putText(frame, f"Recording: {remaining_time:.1f}s", orig=(x1, y1 - 30), color=(0, 0, 255))

            # Detener la grabación después de la duración especificada
            if elapsed_time >= duration:
                record = False
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))


    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)