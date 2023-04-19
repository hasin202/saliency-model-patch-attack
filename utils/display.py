import cv2


def display(window_name, image):
    while True:
        cv2.imshow(window_name, image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


cv2.destroyAllWindows()
