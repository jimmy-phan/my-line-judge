import numpy as np
import time
import cv2
import os


# Log the time
time_start = time.time()
# Start capturing the feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
fgbg = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 120.0, (640, 480))

# image array to store frames
im_array = []


while cap.isOpened():
    ret, frame = cap.read()
    im_array.append(frame)
    out.write(frame)

    fgmask = fgbg.apply(frame)
    cv2.imshow('original', frame)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()