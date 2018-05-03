import cv2
import numpy as np

def go(polys, frame):
    for gesture, pts in polys:
        if gesture == 1:
            color = (255,255,0)
        else:
            color = (0,255,255)

        np_pts = np.array(pts, np.int32)
        cv2.polylines(frame, [np_pts], True, color)
