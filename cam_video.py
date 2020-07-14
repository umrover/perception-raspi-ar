# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import cv2.aruco as aruco
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(2)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    print("Captured Image")
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = np.array(frame.array)
    #print(image.shape)
    #print(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters_create()
    print("Parameters:", parameters)    
    # Lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, 
        aruco_dict, parameters=parameters)
    print("Corners:", corners)
    if(len(corners) > 0):
        gray_img = aruco.drawDetectedMarkers(gray_img, corners, ids, borderColor=(255,0,255))
        cv2.imshow("Detection", gray_img)


    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
    	break
