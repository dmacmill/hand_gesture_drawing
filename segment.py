#! /usr/bin/python
import cv2
import numpy as np

img = cv2.imread("000364.jpg")
imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

skinColorUpper = np.array([20, 0.8 * 255, 0.6 * 255])
skinColorLower = np.array([0, 0.1 * 255, 0.09 * 255])

rangeMask = cv2.inRange(imgHLS, skinColorLower, skinColorUpper)
blurred = cv2.blur(rangeMask, (15,15))
ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('segmented', handmask)
cv2.waitKey(0)
cv2.destroyAllWindows()
