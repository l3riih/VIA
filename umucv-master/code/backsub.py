#!/usr/bin/env python

# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, mkStream

virt = mkStream(dev='dir:../images/cube3.png')
virtbgf = next(virt)

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)
# El kernel se craea con todos uno por que la erosion se tien que aplicar a todos los vecinios por igual.  verifique
# si los píxeles vecinos son ambient iguales a 1 Si es así, entonces el píxel central también se mantiene como 1,
# pero si al menos un vecino es 0, sentences el píxel central se cambia a
kernel = np.ones((3, 3), np.uint8)

update = True

for key, frame in autoStream():

    if key == ord('c'):
        update = not update
        if not update: cv.imshow('background', bgsub.getBackgroundImage())
# Si learningRate es -1, el modelo se actualiza completamente con el fotograma actual.
    # Si learningRate es 0, el modelo no se actualiza.
    # Si learningRate está en el rango (0,1), el modelo se actualiza parcialmente con el fotograma actual.
    fgmask = bgsub.apply(frame, learningRate=-1 if update else 0)
    # Aplicar operaciones de erosión y desenfoque para reducir el ruido y mejorar la segmentación
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    fgmask = cv.medianBlur(fgmask, 3)

    if update: cv.circle(frame, (15, 15), 6, (0, 0, 255), -1)
    cv.imshow('original', frame)
    cv.imshow('mask', fgmask)

    masked = frame.copy()
    masked[fgmask == 0] = 0
    cv.imshow('object', masked)

    # virtbg = next(virt)
    virtbg = virtbgf.copy()
    virtbg[fgmask != 0] = frame[fgmask != 0]
    cv.imshow('virt', virtbg)

cv.destroyAllWindows()
