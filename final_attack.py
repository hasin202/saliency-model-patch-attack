import torch
import cv2
import argparse
from utils.loss_function import SaliencyLoss
from utils.loss_function import *
from utils.data_process import load_input_image, postprocess_img
from utils.load_model import load_model
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
ref_arg1 = parser.add_argument('--target_x', type=int, default=0,
                               help="The x coordinate for the target saliency circle")
args = parser.parse_args()

model = load_model()
model.eval()

if args.target_x > 288 or args.target_x < 0:
    raise argparse.ArgumentError(ref_arg1,
                                 "Make sure that your target x coordinate is less than 288 but greater than 0!")


# # Initialize an empty numpy array to store the modified image
# modified_img = None

# # Set the radius of the circle
radius = 5
brightness = 0.5
x_coordinates, y_coordinates, radius_arr, brightness_arr = [], [], [], []

# # Define the callback function to handle mouse events


def click_event(event, x, y, flags, params):

    # If left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a circle on the modified image
        cv2.circle(img=original_image, center=(x, y), radius=radius,
                   color=(brightness, brightness, brightness), thickness=-1)
        x_coordinates.append(x)
        y_coordinates.append(y)
        radius_arr.append(radius)
        brightness_arr.append(brightness)


# # Read the input image
# target_map = np.zeros((288, 384, 1))

# # Create a window
# cv2.namedWindow('Point Coordinates')

# # Bind the callback function to the window
# cv2.setMouseCallback('Point Coordinates', click_event)

# # Display the image
# cv2.imshow('Point Coordinates', target_map)

# # Copy the input image to the modified image numpy array
# modified_img = np.copy(target_map)

# # Wait for a key press and handle mouse events until ESC key is pressed
# while True:
#     cv2.imshow('Point Coordinates', modified_img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     if k == 43:
#         radius = radius + 10
#     if k == 45 and radius > 0:
#         radius = radius - 10

# # Destroy all windows
# cv2.destroyAllWindows()

# # Print the modified image numpy array
# print(modified_img)


# while True:
#     cv2.imshow('Result', modified_img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()
original_image = load_input_image('./example/5.jpg')
original_image = original_image.squeeze().permute(1, 2, 0).numpy()
original_image = cv2.cvtColor(
    original_image, cv2.COLOR_BGR2RGB)

cv2.namedWindow('Point Coordinates')

# Bind the callback function to the window
cv2.setMouseCallback('Point Coordinates', click_event)


while True:
    cv2.imshow('Point Coordinates', original_image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == 43:
        radius = radius + 1
    if k == 45 and radius > 0:
        radius = radius - 1
    if k == 98 and brightness <= 1:
        brightness = brightness + 0.05
    if k == 100 and brightness >= 0:
        brightness = brightness - 0.05
cv2.destroyAllWindows()

blank = np.zeros((288, 384, 1))

for i in range(len(x_coordinates)):
    if i == 0:
        target_map = cv2.circle(
            blank, (x_coordinates[i], y_coordinates[i]), radius_arr[i], brightness_arr[i], -1)
    else:
        target_map = cv2.circle(
            target_map, (x_coordinates[i], y_coordinates[i]), radius_arr[i], brightness_arr[i], -1)


while True:
    cv2.imshow('target', target_map)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
