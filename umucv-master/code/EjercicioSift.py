#!/usr/bin/env python
import os

# eliminamos muchas coincidencias erróneas mediante el "ratio test"

import cv2 as cv
import time
from umucv.stream import autoStream
from umucv.util import putText


def load_models(folder):
    models = []
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv.imread(os.path.join(folder, file))
            img = cv.resize(img, (400, 300))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            k, d = sift.detectAndCompute(img, mask=None)
            models.append({'name': file, 'img': img, 'keypoints': k, 'descriptors': d})
    return models


sift = cv.SIFT_create(nfeatures=500)
matcher = cv.BFMatcher()
models = load_models('siftFotos/')


def find_best_model(frame):
    best_count = 0
    best_model = None
    good = None
    for model in models:
        matches = matcher.knnMatch(frame, model['descriptors'], k=2)

        good = []
        for m in matches:
            if len(m) >= 2:
                best, second = m
                if best.distance < 0.75 * second.distance:
                    good.append(best)

        if len(good) > best_count:
            best_count = len(good)
            best_model = model

    return best_model, best_count, good


x0 = None

for key, frame in autoStream():

    if key == ord('x'):
        x0 = None

    resizeF = cv.resize(frame, (400, 300))

    t0 = time.time()
    keypoints, descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000 * (t1 - t0):.0f} ms')

    if descriptors is None:
        continue

    bestModel, bestCount, good = find_best_model(descriptors)

    if bestModel is not None:
        x0 = frame
        # Calcular porcentaje de coincidencias
        match_percentage = bestCount / len(bestModel["keypoints"]) * 100
        print("best match " + str(match_percentage))

        # Rechazar la decisión si el porcentaje de coincidencias es menor a un umbral (por ejemplo, 10%)
        if match_percentage < 10:
            bestModel = None
            x0 = None

    if x0 is None:
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.drawKeypoints(frame, keypoints, frame, color=(100, 150, 255), flags=flag)
        cv.imshow('SIFT', frame)
    else:
        print("Entra")
        imgm = cv.drawMatches(frame, keypoints, bestModel["img"], bestModel["keypoints"], good,
                              flags=0,
                              matchColor=(128, 255, 128),
                              singlePointColor=(128, 128, 128),
                              outImg=None)
        t2 = time.time()
        putText(imgm, f'{len(good)} matches  {1000 * (t2 - t1):.0f} ms',
                orig=(5, 36), color=(200, 255, 200))
        putText(imgm, f'Mejor modelo: {bestModel["name"]} ({match_percentage:.2f}%)',
                orig=(5, 72), color=(200, 255, 200))
        cv.imshow("SIFT", imgm)

'''
#!/usr/bin/env python
import os

import cv2 as cv
import time
from umucv.stream import autoStream
from umucv.util import putText

directorio = 'siftFotos/'  # Ruta del directorio deseado
archivos = os.listdir(directorio)
rutas_archivos_ordenadas = sorted([os.path.join(directorio, archivo) for archivo in archivos])


def readrgb(file):
    return cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)


def readfiles():
    return [readrgb(file) for file in rutas_archivos_ordenadas]


imgs = readfiles()
sift = cv.SIFT_create(nfeatures=500)
dis = [sift.detectAndCompute(x, None) for x in imgs]

matcher = cv.BFMatcher()

x0 = None

for key, frame in autoStream():

    if key == ord('x'):
        x0 = None

    t0 = time.time()
    keypoints, descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000 * (t1 - t0):.0f} ms')

    if key == ord('c'):
        k0, d0, x0 = keypoints, descriptors, frame

    if x0 is None:
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.drawKeypoints(frame, keypoints, frame, color=(100, 150, 255), flags=flag)
        cv.imshow('SIFT', frame)
    else:
        t2 = time.time()
        # solicitamos las dos mejores coincidencias de cada punto, no solo la mejor
        matches = matcher.knnMatch(descriptors, d0, k=2)
        t3 = time.time()

        # ratio test
        # nos quedamos solo con las coincidencias que son mucho mejores que
        # que la "segunda opción". Es decir, si un punto se parece más o menos lo mismo
        # a dos puntos diferentes del modelo lo eliminamos.
        good = []
        for m in matches:
            if len(m) >= 2:
                best, second = m
                if best.distance < 0.75 * second.distance:
                    good.append(best)

        imgm = cv.drawMatches(frame, keypoints, x0, k0, good,
                              flags=0,
                              matchColor=(128, 255, 128),
                              singlePointColor=(128, 128, 128),
                              outImg=None)

        putText(imgm, f'{len(good)} matches  {1000 * (t3 - t2):.0f} ms',
                orig=(5, 36), color=(200, 255, 200))
        cv.imshow("SIFT", imgm)
'''
