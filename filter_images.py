#! Python

from __future__ import (
    division, absolute_import, print_function, unicode_literals)

import cv2 as cv
import numpy as np

def white_bal(file_in="test.jpg", name="output.jpg"):

	# Insert any filename with path
	img = cv.imread(file_in)
	final = cv.cvtColor(img, cv.COLOR_BGR2LAB)

	avg_a = np.average(final[:, :, 1])
	avg_b = np.average(final[:, :, 2])
	
	for x in range(final.shape[0]):
	    for y in range(final.shape[1]):
	        l, a, b = final[x, y, :]
	        # fix for CV correction
	        l *= 100 / 255.0
	        final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
	        final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
	
	final = cv.cvtColor(final, cv.COLOR_LAB2BGR)
	final = np.hstack((img, final))

	cv.imwrite(name, final)

white_bal()

