
import os
import cv2
import numpy as np
import colorsys
import imutils

def rgb_hue(r,g,b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0,0,0]


# path to test file
VIDEO_FILE = os.path.join("videos", "test2.mp4")
is_file = os.path.isfile(VIDEO_FILE)
if not is_file:
    print("path to file is invalid")
else:
    print("video file exists")

# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# set up target color and ranges
r,g,b = (163, 186, 66)
h = rgb_hue(r, g, b)

hue_threshold = 5

# define range of green color in HSV
upper_green = np.array([h + 5, 255, 255])
lower_green = np.array([h - 5, 100, 100])

cap = cv2.VideoCapture(VIDEO_FILE)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(1):
    # Take each frame
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=640)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # these display the different views
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

#    if ret==True:

 #       out.write(mask)
                 
 #       cv2.imshow('mask',mask)
  #  else: 
 #       break

    k = cv2.waitKey(50) & 0xFF
    if k == ord("q"):
        break
cap.release()
#out.release()
cv2.destroyAllWindows()

#print(flags)
