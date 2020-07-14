'''
This is a script to process an rpi cam v2 stream
and then send back coordinates (in the global 
coordinate frame) of any identified AR tags
'''

from io import BytesIO
from time import sleep
from picamera import PiCamera

my_file = open('my_image.jpg', 'wb')
# stream = BytesIO() 
cam = PiCamera()
cam.resolution = (1024,768)
cam.start_preview()
#camera warm up time
sleep(2)
cam.capture(my_file)
#do some stuff

my_file.close()

