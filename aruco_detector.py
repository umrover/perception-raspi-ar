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
from imutils.video import VideoStream
import imutils
import yaml

def main():

    # my_file = open('my_image.jpg', 'wb')
    # # stream = BytesIO()
    # cam = PiCamera()
    res = (320,240)
    fps = 30

    vs = VideoStream(src=0, usePiCamera=True, resolution=res, 
        framerate=fps).start()

    #camera warm up time
    time.sleep(2)
    

    # cam = PiCamera()
    # cam.resolution = (1024,768)
    # cam.start_preview()
    # #camera warm-up
    # time.sleep(2)

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print(frame.shape)

        # Our operations on the frame come here
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = aruco.DetectorParameters_create()
        print("Parameters:", parameters)

        # Lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, 
            aruco_dict, parameters=parameters)
        print("Corners:", corners)

        if(len(id) > 0):
            for idx, i in enumerate(id):
                # call function to calculate ar tag heading relative to rover
                heading = global_coord_trans(id[i], corners[i*4:(i+1)*4])
                # this is where we would send heading over LCM channel
                # also note global coord trans could happen somewhere 
                #    outside of this script
        gray_img = aruco.drawDetectedMarkers(gray_img, corners)
        #print(rejectedImgPoints)
        
        # Display the resulting frame
        cv2.imshow('frame',gray_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


    return


def global_coord_trans(id, corners):
    #transform the corners into an image coordinate system
    #representation, then transform into the global
    #coordinate frame of the rover using yaml file with
    #extrinsics

    #use corners detected to find center pixel
    center = np.array([(corners[0] + corners[1]) / 2], 
        [(corners[2] + corners[3]) / 2], [1])

    extrinsics = file("right_cam_extrinsic.yaml", 'r')
    cam_matrix = file("right_cam_intrinsic.yaml", 'r')
    try:
        extrn = yaml.safe_load(extrinsics)
        intrn = yaml.safe_load(cam_matrix)

        rotation = extrn['rotation']
        translation = extrn['translation']

        A = intrn['cam_matrix'] #this is given by cam calibration tool

        inner_mat = np.matmul(np.dot(s, center), np.linalg.inv(A)) - translation #missing s for scaling pixel coords. tbh idk what it is
        global_coords = np.matmul(np.linalg.inv(rotation), 


    except yaml.YAMLError as exc:
        print("Failed global transformation for", id, "ERROR:", exc)

    return heading


if __name__ == '__main__':
    main()
