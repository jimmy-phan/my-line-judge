import time
import cv2
import os
import numpy as np
import imutils

def rgb_hue(r,g,b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0,0,0]


# set up target color and ranges
r, g, b = (163, 186, 66)
h = rgb_hue(r, g, b)

hue_threshold = 10

# define range of green color in HSV
upper_green = np.array([h + 5, 255, 255])
lower_green = np.array([h - 5, 100, 100])

# image = cv2.imread("C:/Users/jppha/Documents/masters-repository/my-line-judge/videos/points.jpg",0)


VIDEO_FILE = os.path.join("videos", "output1.avi")
is_file = os.path.isfile(VIDEO_FILE)
if not is_file:
    print("path to file is invalid")
else:
    print("video file exists")

# cap = cv2.VideoCapture("/Users/jppha/Documents/masters-repository/my-line-judge/videos/output.avi")

cap = cv2.VideoCapture(VIDEO_FILE)
# parameters for the ShiTomasi corner detector
feature_params = dict(maxCorners = 1000,
                          qualityLevel  = 0.3,
                          minDistance = 7,
                          blockSize = 7 )

# parameters for the lucas kanade optical flow
lk_params = dict( winSize = (15,15),
                       maxLevel =2,
                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

ret, first_frame = cap.read()
first_frame = imutils.resize(first_frame, width=640)
# cv2.imshow("first", first_frame)
gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray_frame)

# cv2.waitKey(0)

# hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, lower_green, upper_green)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)

# p0 = cv2.goodFeaturesToTrack(gray_frame, mask = mask, **feature_params)

mask1 = np.zeros_like(first_frame)

while(1):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    gray_frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gray_frame2 = cv2.bitwise_and(frame,frame, mask= mask)

    # cv2.imshow('gray_frame2',gray_frame2)
    # cv2.imshow('mask', mask)

    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break
    p0 = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
    temp = p0

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frame, gray_frame2, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b, = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask1, (a,b), (c,d), color[i].tolist(),2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask1)

        cv2.imshow('frame', img)
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
            break

        # Now update the previous frame and previous points
        gray_frame = gray_frame2.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()