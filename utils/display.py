import cv2
import numpy as np


def display(window_name, image):
    while True:
        cv2.imshow(window_name, image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
