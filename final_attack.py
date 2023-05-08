import torch
import cv2
import argparse
from utils.loss_function import *
from utils.data_process import load_input_image, form_target_map, form_target_map_mask, postprocess_img
from utils.load_model import load_model
from utils.display import display
from utils.attack import attack
import numpy as np
import os
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

ref_arg1 = parser.add_argument('--loss_target', type=float, default=0.5,
                               help="The goal loss function value")
ref_arg2 = parser.add_argument('--lr', type=int, default=1,
                               help="The goal loss function value")
ref_arg3 = parser.add_argument('--img', type=str, default="./example/1.jpg",
                               help="The path to the image you would like to perform the attack on")
ref_arg4 = parser.add_argument('--mask', type=str, default=True,
                               help="True for only updating pixels in the target region, False for updating all pixels")
args = parser.parse_args()

model = load_model()
model.eval()

radius = 10
brightness = 0.5
x_coordinates, y_coordinates, radius_arr, brightness_arr = [0], [0], [0], [0]


# Define the callback function to handle mouse events
def click_event(event, x, y, flags, params):

    # If left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a circle on the modified image
        cv2.circle(img=modified_image, center=(x, y), radius=radius,
                   color=(brightness, brightness, brightness), thickness=-1)
        x_coordinates.append(x)
        y_coordinates.append(y)
        radius_arr.append(radius)
        brightness_arr.append(brightness)
        print(brightness)


# Load the input image and process it to be displayed
original_image = load_input_image(args.img)
c = load_input_image("./example/6.jpg")
print(original_image.dtype)
modified_image = np.copy(original_image)

cv2.namedWindow('original image')

# Bind the callback function to the window
cv2.setMouseCallback('original image', click_event)

while True:
    cv2.imshow('original image', modified_image)
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


target_map = form_target_map(
    original_image, x_coordinates, y_coordinates, radius_arr, brightness_arr)
target_map_mask = form_target_map_mask(
    original_image, x_coordinates, y_coordinates, radius_arr, brightness_arr)
target_map_mask = target_map_mask.astype(np.uint8)

save_target_map = (target_map*255)
cv2.imwrite(os.path.basename(os.path.normpath(
    args.img)), save_target_map)
display('target', target_map)
display("mask", target_map_mask)

original_image = torch.from_numpy(original_image)
original_image = original_image.type(torch.FloatTensor).to(device)
original_image = torch.permute(original_image, (2, 0, 1)).unsqueeze(0)

target_map = torch.from_numpy(target_map)
target_map = target_map.unsqueeze(0).unsqueeze(0)


final_patch, iterations, output = attack(model,
                                         original_image, target_map, target_map_mask, args.loss_target, args.lr, args.mask)


final_patch = postprocess_img(
    final_patch.squeeze().detach().permute(1, 2, 0), args.img)
output = postprocess_img(
    output.squeeze().unsqueeze(0).detach().permute(1, 2, 0), args.img)


display('final', final_patch)
display('output', output)
