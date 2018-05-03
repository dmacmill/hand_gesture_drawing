# hand_gesture_drawing

## File Descriptions

### segment.py
Iterates over source dir (link to the source dir in global READDIR) and returns black and white masks of size 64 x 64px 
in the SAVEDIR. Mask should return all blobs that could be a human hand - this is mean to take raw RGB data images and
turns them into images that can be easily consumed by the ML algorithm.

### hand_segmentation.py
Merly a test of how one could perform gesture recognition without any ML algorithm, instead opting for an analysis of
the contours of the hand, how some fingers are up and some are down, to see how one could use this to perform basic
drawing feature.

### display.py
Opens the webcam and applys filter found in hand_segmentation
