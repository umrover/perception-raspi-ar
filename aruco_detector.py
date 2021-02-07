'''
This program detects aruco ar tags from
a camera stream

Next steps: perform transformation from
image coordinate system to global coordinate
system using extrinsic matrix
'''

import cv2
import cv2.aruco as aruco
import numpy as np
from io import BytesIO
import time
import os
from picamera import PiCamera
from picamera.array import PiRGBArray
from imutils.video import VideoStream
import imutils
import yaml

### For the print statements, feel free to put them in #IFDEBUG flags

def main():

    # establish global constants (hardcode in file or make optional param to fn call)
    # KNOWN_WIDTH is the width of the AR tags to detect in URC competition
    KNOWN_WIDTH = 20 # unit is cm    

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    # allow the camera to warmup
    time.sleep(2)
    
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # print("Captured Image")
        # grab the raw NumPy array representing the image
        image = np.array(frame.array)

        # Our operations on the frame come here
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #TODO: change "cv2.aruco.DICT_5x5" to rover aruco lib
        aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = aruco.DetectorParameters_create()
        # print("Parameters:", parameters)

        # Lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, 
            aruco_dict, parameters=parameters)
        # print("Corners:", corners)
        # print(type(id))

        if(len(corners) > 0):
            for i in range(len(corners)): # used to be id                
                # call function to calculate ar tag heading relative to rover
                depth = get_depth(i, corners[i]) #used to be id[i]
                heading = global_coord_trans(i, corners[i], depth) # id[i]
                # this is where we would send heading over LCM channel
                # note: global coord trans could happen somewhere 
                #       outside of this script. It doesn't, but it could
                print("Depth and Heading of", i, "are:", depth, "&", heading)
        gray_img = aruco.drawDetectedMarkers(gray_img, corners)
        #print(rejectedImgPoints)
        
        # Display the resulting frame
        cv2.imshow('frame',gray_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rawCapture.truncate(0)

    # When everything done, release the capture
    # cap.release()
    cv2.destroyAllWindows()


    return


def global_coord_trans(id, corners, depth):
    #transform the corners into an image coordinate system
    #representation, then transform into the global
    #coordinate frame of the rover using yaml file with
    #extrinsics

    #use corners detected to find center pixel
    center = np.array([(corners[0][0][1] + corners[0][3][1]) / 2, 
        (corners[0][1][0] + corners[0][2][0]) / 2])

    with open("cam_intrinsic.yaml", 'r') as cam_matrix:
    #cam_matrix = file("cam_intrinsic.yaml", 'r')
    # extrinsics = file("cam_extrinsic.yaml", 'r')
        try:
            # extrn = yaml.safe_load(extrinsics)
            intrn = yaml.safe_load(cam_matrix)

            # rotation = extrn['rotation']
            # translation = extrn['translation']

            A = intrn['cam_matrix']             #this is given by cam calibration tool
            A = np.array(A).reshape((3, 3))     #reshape data to 3x3 matrix

            # inner_mat = np.matmul(np.dot(s, center), np.linalg.inv(A)) - translation #missing s for scaling pixel coords. tbh idk what it is
            # global_coords = np.matmul(np.linalg.inv(rotation), 0) #change the 0 at the end 

            u = center[1]
            v = center[0] # 0 and 1 may be mixed up
            u0 = A[0,2]
            v0 = A[1,2]
            fx = A[0,0]
            fy = A[1,1]

            x = (u - u0) * depth / fx
            y = (v - v0) * depth / fy
            z = depth

        except yaml.YAMLError as exc:
            print("Failed global transformation for", id, "ERROR:", exc)

    # (x, y, z) are 3D backprojected coordinates. Points relative to camera frame
    # Another transformation is needed between camera frame and rover frame (using extrinsics)
    return (x, y, z) # place heading back in


def get_depth(id, corners):
    # print(corners)
    # print(corners.shape)
    # unit of width is pixels
    width = abs(((corners[0][2][1] + corners[0][3][1]) / 2) - ((corners[0][0][1] + corners[0][1][1]) / 2))
    # print(width)
    
    with open("cam_intrinsic.yaml", 'r') as cam_matrix:
    #cam_matrix = file("cam_intrinsic.yaml", 'r')
        try:
            intrn = yaml.safe_load(cam_matrix)

            A = intrn['cam_matrix']             #this is given by cam calibration tool
            A = np.array(A).reshape((3, 3))     #reshape data to 3x3 matrix

            fx = A[0,0]
            fy = A[1,1]

            FOCAL_LENGTH = (fx + fy) / 2.0      #probably wrong
            KNOWN_WIDTH = 20

        except yaml.YAMLError as exc:
            print("Failed focal length loading for", id, "ERROR:", exc)

    depth = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, width) / 10
    print("depth is:", depth, "\t inches")
    return depth


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
    # this makes use of a triangle similarity
	return (knownWidth * focalLength) / perWidth


if __name__ == '__main__':
    main()
