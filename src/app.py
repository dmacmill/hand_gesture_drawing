

import cv2

import process
import ml_model
import draw
import numpy as np


# get camera
cap = cv2.VideoCapture(0)

pts = []
polys = []
last_gesture = 0
lost_gesture = 0

count = 0
num_empty = 30

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    count += 1
    # process frame w/ ML
    img_data, mask, point = process.process(frame)

    if  len(img_data) == 64:

        gesture = ml_model.predict(img_data)
        
        print "***************************************"
        print ""
        print gesture
        print "" 
        print pts 
        print "" 
        print polys 
        print "" 
        print "***************************************"
        
        if gesture > 0:
            last_gesture = gesture
            pts.append(point)
        else: 
            lost_gesture += 1
            if lost_gesture > num_empty:
                lost_gesture = 0
                polys.append(( last_gesture, pts ))
                pts = []
            else:
                print num_empty-lost_gesture + " blank frames left until we draw"

        draw.go(polys, frame)
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
