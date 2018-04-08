#! /usr/bin/python


import cv2
import numpy as np
import hand_segmentation as hs

# get camera
cap = cv2.VideoCapture(0)

count = 0
pts = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #frame = hs.process_image(frame)
    hs.process_image(frame)
    
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # draw stuff
    count += 1
    current_gesture = count % 30
    pts.append([count, current_gesture])
    np_pts = np.array(pts, np.int32)
    cv2.polylines(frame, [np_pts], True, (255,255,0))
    cv2.putText(frame, str(current_gesture), (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)   
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






