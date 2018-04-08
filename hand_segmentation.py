#! /usr/bin/python

import cv2
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

# TODO:
# wrap code in a class / make more modular
# find hull defects with cv2.convexityDefects(contours, cc)

def segment(imgHLS):
    # Segmentation:
    skinColorUpper = np.array([20, 0.8 * 255, 0.6 * 255])
    skinColorLower = np.array([0, 0.1 * 255, 0.09 * 255])

    rangeMask = cv2.inRange(imgHLS, skinColorLower, skinColorUpper)
    blurred = cv2.blur(rangeMask, (15,15))
    ret, handmask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow('segmented', handmask)
    return handmask

def get_contours(handmask):
    _, contours, hierarchy = cv2.findContours(
        handmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    index_current_largest = 0
    i = 0
    for c in contours:
        if(c.size > contours[index_current_largest].size):
            index_current_largest = i
        i+=1
    return contours, index_current_largest

def group_clusters(roughHull):
    # AffinityPropagation algorithm:
    af = AffinityPropagation(preference=-100).fit(roughHull.squeeze(1))
    print "Fingers up: " + str(len(af.cluster_centers_indices_) - 2)
    cluster_indicators = af.fit_predict(roughHull.squeeze(1))
    
    # map the list that the affinity provided to the current roughHull, then
    # find the mean point of all of them
    cluster_centers = []
    current_cluster = 0
    while(current_cluster < len(af.cluster_centers_indices_)):
        i=0
        cluster=[]
        for x in cluster_indicators:
            if(x == current_cluster):
                cluster.append(roughHull[i][0].tolist())
            i+=1
        # now find the average point between these
        average = [0, 0]
        for y in cluster:
            average = [average[0] + y[0], average[1] + y[1]]
        average = [average[0]/len(cluster), average[1]/len(cluster)]
        cluster_centers.append(np.array([average]))
        current_cluster += 1
    cc = np.array(cluster_centers)
    return cc

def process_image(img):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Segmentation:
    handmask = segment(imgHLS)
    
    # Get Contour: find the largest contour of the hand mask to ensure no
    # white noise is interfering
    contours, i = get_contours(handmask)
    cv2.drawContours(img, contours, i, (0,255,0), 3)
    
    # Get hull: turn the contour into a polygonal shape that will have
    # verticies at the tips of the fingers.
    print i
    if (len(contours) > 0):
        roughHull = cv2.convexHull(contours[i])
        cv2.drawContours(img, roughHull, -1, (0,0,255), 3)
    
    # group the clusters together and find the mean point for each
        convexHull = group_clusters(roughHull)
        cv2.drawContours(img, convexHull, -1, (0,0,0), 3)
    
    #print contours[i]
    #print convexHull

    # convexHull needs to return indicies for this part in order for
    # convexityDefects to notice, so we call convexHull again with
    # returnPoints = False
        indexHull = cv2.convexHull(contours[i], returnPoints = False)
        defects = cv2.convexityDefects(contours[i], indexHull)

        for j in range(defects.shape[0]):
            s,e,f,d = defects[j,0]
            start = tuple(contours[i][s])
            end = tuple(contours[i][e])
            far = tuple(contours[i][f])
            
            # far[0] points should be compared to the points in convexHull 
            # due to the fact that we've just duplicated a bunch of points
            # from the returnPoints snafu
            is_repeat = False
            for hullpoint in convexHull:
                diff_x = abs(hullpoint[0][0] - far[0][0])
                if(diff_x <= 10):
                    diff_y = abs(hullpoint[0][1] - far[0][1])
                    if(diff_y <= 10):
                        is_repeat = True
            if(is_repeat == False):
                convexHull = np.insert(convexHull, 0, far[0], axis=1)
                #cv2.circle(img, tuple(far[0]), 5, [255,255,0], -1)
        
    return handmask
    #print roughHull.squeeze(1).tolist() #return like this
    #cv2.drawContours(img, convexHull, -1, (0,0,0), 3)
    
    #cv2.imshow('contours', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

#if(__name__ == "__main__"):
#    main()
