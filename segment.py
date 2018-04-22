#! /usr/bin/python
import os
import cv2
import numpy as np

SAVEDIR = "processed/masked_21"

def mask(img):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    skinColorUpper = np.array([20, 0.8 * 255, 0.6 * 255])
    skinColorLower = np.array([0, 0.1 * 255, 0.09 * 255])
    
    rangeMask = cv2.inRange(imgHLS, skinColorLower, skinColorUpper)
    blurred = cv2.blur(rangeMask, (15,15))
    ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    return handmask

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

for filename in os.listdir("/processed/"):
    if(filename.endswith(".jpg")):
        img = cv2.imread("/home/daniel/Documents/CV/hand_gesture_drawing/processed/hand_gesture_21" + filename, 1)
        handmask = mask(img)
        cv2.imwrite(SAVEDIR + filename, handmask)
