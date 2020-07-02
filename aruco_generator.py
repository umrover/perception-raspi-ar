'''
# This file is an aruco tag generator
# Images are saved in ./aruco_tags
'''

import cv2
import numpy as np

def main(DEBUG=False):
    #load the aruco dictionary
    marker = np.zeros((500, 500))
    if DEBUG:
        print(marker.shape)
        print(type(marker))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    
    #for loop to generate the markers from 0 to var and save the images
    for i in range(0, 10):
        # TODO: make range start and end a required cli variable
        # TODO: add checking to see if the markers exist & 
                # overwrite existing imgs
        marker = cv2.aruco.drawMarker(dictionary, i, marker.shape[0], marker, 1)
        cv2.imwrite("aruco_tags/marker" + str(i) + ".png", marker)
        if DEBUG:
            print("sum of the matrix:\t", np.sum(marker))

    print("Done generating markers!")



if __name__ == '__main__':
    main()
