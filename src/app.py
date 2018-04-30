

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
    gesture = ml_model.predict(img_data)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if mask is not None:
        cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
