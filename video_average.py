import numpy as np
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

# image array to store frames
im_array = []

fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    im_array.append(frame)
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()