import numpy as np
# from imutils.video import WebcamVideoStream
from imutils.video import FPS
from FPS import FPS
from WebcamVideoStream import WebcamVideoStream
import argparse
import imutils
import cv2

# video = cv2.VideoCapture(0)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# video.set(cv2.CAP_PROP_FPS, 120)

video = WebcamVideoStream(src=0).start()
fps = FPS().start()


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 120.0, (640, 480))

while(video.isOpened()):
    ret, frame = video.read()
    if ret==True:
        # frame = cv2.flip(frame, 0)
        #
        # # write the flipped frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release everything if job is finished
video.release()
out.release()
cv2.destroyAllWindows()