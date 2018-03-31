import numpy as np
import imutils
import time
import os
import vid_to_frames
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Our operations on the frame come here

    # Display the resulting frame
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(cv2.__version__)
# video = cv2.VideoCapture('output.avi')
# success,image = video.read()
# count = 0
# success = True
# while success:
#     success,image = video.read()
#     print('Read a new frame: ', success)
#     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#     count += 1


# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()