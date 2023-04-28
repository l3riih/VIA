
import cv2 as cv
import numpy as np
from collections import deque
from umucv.stream import autoStream
colors = deque(maxlen=3)



def subtr(arr, num):
    result = np.empty_like(arr)
    for i in range(len(arr)):

        if  arr[i] - num < 0:
            result[i] = 0
        else:
            result[i] = arr[i] - num
        print(result)
    return result

def on_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        pixel = frame[y, x]
        hsv_pixel = cv.cvtColor(np.uint8([[pixel]]), cv.COLOR_BGR2HSV)[0][0]
        colors.append((0 if hsv_pixel[0] - 10 < 0 else hsv_pixel[0] - 10, 255 if hsv_pixel[0] + 10 > 255 else hsv_pixel[0] + 10))





cv.namedWindow("Video")
cv.setMouseCallback("Video", on_mouse_click)

for key, frame in autoStream():

    if len(colors) == 3:
        masks = []

        for color in colors:
            lower_bound = np.array([color[0], 100, 100], dtype=np.uint8)
            upper_bound = np.array([color[1], 255, 255], dtype=np.uint8)

            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv_frame, lower_bound, upper_bound)
            masks.append(mask)
            # Aplicamos un desenfoque gaussiano para reducir el ruido
            mask = cv.GaussianBlur(mask, (5, 5), 0)

            # Aplicamos operaciones morfológicas para mejorar la detección de objetos
            kernel = np.ones((5, 5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            masks.append(mask)


        combined_mask = cv.bitwise_or(masks[0], masks[1])
        combined_mask = cv.bitwise_or(combined_mask, masks[2])

        output = cv.bitwise_and(frame, frame, mask=combined_mask)

        # Aplicamos componentes conexos en la máscara combinada
        nnum_objects, labels, stats, _ = cv.connectedComponentsWithStats(combined_mask)
        min_area = 100  # Puedes ajustar este valor según el tamaño mínimo deseado
        filtered_stats = [stat for stat in stats if stat[cv.CC_STAT_AREA] > min_area]
        num_filtered_objects = len(filtered_stats) - 1

        cv.putText(output, f"Objetos encontrados: {num_filtered_objects}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow("Resultado", output)

    cv.imshow("Video", frame)


cv.destroyAllWindows()

