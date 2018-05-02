

import cv2

import process
import ml_model

import numpy as np


# get camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # process frame w/ ML
    img_data, mask = process.process(frame)

    if  len(img_data) == 64:
        gesture = ml_model.predict(img_data)
        print "***************************************"
        print ""
        print gesture
        print ""
        print "***************************************"
    else:
        print "img_data is too short: " + str(len(img_data))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if mask is not None:
        cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
