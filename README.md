# Performing adversarial patch attacks on a saliency model <br>

This repository provides the Pytorch implementation for performing an adversarial patch attack on a saliency model (TranSalNet) using methods from the paper Adversarial Patch which performed attacks on an image classification model.<br>

All of the generated stimuli that were shown to participants can be found in the folder: generated_stimuli_for_study

## Requirements

- Python 3.9.12
- Pytorch 1.13.1
- Torchvision 0.14.1
- OpenCV-Python 4.7.0.72
- SciPy 1.6.0
- tqdm 4.56.0
- matplotlib 3.7.1
- scikit-image 0.20.0
- pandas 1.5.3

## Setup:

TranSalNet is the saliency model that is attacked in this paper so please follow the steps to setup the model from this readme: https://github.com/LJOVO/TranSalNet/blob/master/README.md

## Performing an attack:

To perform an attack the file: final_attack.py can be run. This file takes a number of command line arguments to determine the nature of the attack:

- --img: Sets the image to be attacked
- --lr: sets the learning rate
- --loss_target: sets the threshold for loss, once loss is above this the attack will be finished and the output will be displayed.
- --mask: Determines what variation of the attack will be used. If this is set to True then the mask variation will be used. If it is False then the standard variation will be used.

An example of a command to run the file:

```
python3 final_attack.py --img ./example/trainSet/Stimuli/Action/019.jpg --lr 10 --loss_target 0.5 --mask False
```

This sets the learning rate to 10, loss target to 0.5 and uses the standard variation of the attack.

Once the command is run a window displaying the image passed in will be displayed. You can draw a target saliency map on this image by simply clicking on it with you mouse. The controls to the UI are:

- Pressing "b" makes the circles brighter
- Pressing "d" makes the circles dimmer
- Pressing "+" increases the radius of the circle
- Pressing"-" decreases the radius of the circle

Once you are happy with your target map press "esc" twice, and the attack will begin. Once completed the final image will be displayed in a window on your screen. Pressing "esc" one last time will show TranSalNets saliency output for the final image.
