#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time

from umucv.util import ROI, putText
from umucv.stream import autoStream


cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")

# Inicializar sustractor de fondo
fgbg = cv.createBackgroundSubtractorMOG2()

# Variables para controlar la grabación
recording = False
start_time = 0
duration = 3  # Duración de la grabación en segundos
threshold = 500
significativo = 0
for key, frame in autoStream():
    if region.roi:
        [x1, y1, x2, y2] = region.roi

        # Aplicar sustracción de fondo a la ROI
        roi_frame = frame[y1:y2 + 1, x1:x2 + 1]
        fgmask = fgbg.apply(roi_frame)
        cv.imshow("fgmask", fgmask)

        # Obtener el área de la máscara de diferencia
        move = np.sum(fgmask > 10)

        if move > threshold:
            significativo +=1
        else:
            significativo =  0

            # Si se detecta movimiento, iniciar la grabación
        if significativo > 3:
            if not recording:
                recording = True
                start_time = time.time()

        if recording:
            # Si se está grabando, mostrar el tiempo restante
            elapsed_time = time.time() - start_time
            remaining_time = max(0, duration - elapsed_time)
            putText(frame, f"Recording: {remaining_time:.1f}s", orig=(x1, y1 - 30), color=(0, 0, 255))

            # Detener la grabación después de la duración especificada
            if elapsed_time >= duration:
                recording = False

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))

    h, w, _ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input', frame)

    if key == ord('x'):
        region.roi = []

    if key == ord('q'):
        break

cv.destroyAllWindows()
