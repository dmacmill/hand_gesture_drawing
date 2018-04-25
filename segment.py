#! /usr/bin/python
import os
import cv2
import numpy as np

BASEDIR = "/home/daniel/Documents/CV/hand_gesture_drawing/"
READDIR = "processed/hand_gesture_21/"
SAVEDIR = "processed/masked_21/"

# Crop, Resize, Save as bmp

def mask(img):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    skinColorUpper = np.array([25, 0.8 * 255, 0.6 * 255])
    skinColorLower = np.array([0, 0.1 * 255, 0.09 * 255])
    
    rangeMask = cv2.inRange(imgHLS, skinColorLower, skinColorUpper)
    blurred = cv2.blur(rangeMask, (15,15))
    ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    return handmask

def get_square(image,square_size):

    height,width,depth=image.shape[0], image.shape[1], -1
    if(len(image.shape) == 3):
        depth = image.shape[2]
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4
    mask = None

    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    if(depth == -1):
        mask = np.zeros((differ,differ), dtype="uint8")   
        mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    else:
        mask = np.zeros((differ,differ, depth), dtype="uint8")   
        mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width, depth]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)

    return mask

def crop(handmask):
    dst = np.zeros((handmask.shape[0], handmask.shape[1], 0), dtype="uint8")
    _,contours,hierarchy = cv2.findContours(
        handmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    index_current_largest = 0
    i = 0
    for c in contours:
        if(c.size > contours[index_current_largest].size):
            index_current_largest = i
        i+=1
    if(len(contours) == 0):
        return handmask
    #return contours, index_current_largest
    br=cv2.boundingRect(contours[index_current_largest])
    print br
    print contours[index_current_largest].shape
    print dst.shape
    crop_img = handmask[br[1]:br[1]+br[3] , br[0]:br[0]+br[2]]
    #cv2.drawContours(dst, contours, index_current_largest, (255,255,255), 3)
    return get_square(crop_img, 64)

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

for filename in os.listdir(READDIR):
    if(filename.endswith(".jpg")):
        print filename
        img = cv2.imread(BASEDIR + READDIR + filename, 1)
        handmask = mask(img)
        cropmask = crop(handmask)
        cv2.imwrite(SAVEDIR + filename, cropmask)
