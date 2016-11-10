from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import socket
import sys

sys.path.append("car_dir/")

#import video_dir
#import car_dir
#import motor

video_output = True

from time import time
time_next  = time()

if video_output:
  time_delay = 0.2
else:
  time_delay = 0.05

import train

udp_host = "localhost"
udp_port = 4000
sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

key_wait_time = 10
input_size = 128

capture = cv2.VideoCapture(0)

model = train.build_model()
model.load('checkpoints/road_model1-200000')

def process_frame(frame):
  pr = model.predict(frame[np.newaxis, :, :, np.newaxis])
  return pr[0][0]*10

while capture.isOpened():
  success, frame = capture.read()

  if success:
    time_now = time()
    if time_now >= time_next:
      time_next = time_now + time_delay

      frame_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame_small = cv2.resize(frame_gray, (128, 128))
      
      if video_output:
        cv2.imshow('video', frame_small)

      output_value =  process_frame(frame_small)
      print(output_value)
      if output_value > 0:
		    pass#car_dir.turn_left()
      elif output_value < 0:
		    pass#car_dir.turn_right()
      else:
		    pass#car_dir.home()

      #motor.forward()
      #sock.sendto('ai: %.6f' % output_value, (udp_host, udp_port))

    ch = cv2.waitKey(key_wait_time) & 0xFF
    if ch == 27:
      break
    if ch == ord('q'):
      break

capture.release()
cv2.destroyAllWindows()
