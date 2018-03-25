#! /usr/bin/python

import cv2
import numpy as np

img = cv2.imread('hand.jpg')
imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# Segmentation:
skinColorUpper = np.array([20, 0.8 * 255, 0.6 * 255])
skinColorLower = np.array([0, 0.1 * 255, 0.09 * 255])

rangeMask = cv2.inRange(imgHLS, skinColorLower, skinColorUpper)
blurred = cv2.blur(rangeMask, (15,15))
ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
#cv2.imshow('segmented', handmask)

# Get Contour: find the largest contour of the hand mask to ensure no
# white noise is interfering
_, contours, hierarchy = cv2.findContours(
    handmask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
i = 0
for c in contours:
    if(c.size > contours[i].size):
        i = contours.index(c)
cv2.drawContours(img, contours, i, (0,255,0), 3)
#cv2.imshow('contours', img)

# Get hull: turn the contour into a polygonal shape that will have
# verticies at the tips of the fingers.
roughHull = cv2.convexHull(contours[i])
cv2.drawContours(img, roughHull, -1, (0,0,255), 3)
print type(roughHull)

cv2.imshow('contours', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
