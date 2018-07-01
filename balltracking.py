# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import os


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
r, g, b = (237, 241, 42)
hue_threshold = 5
h = rgb_hue(r, g, b)
upper_green = np.array([h + hue_threshold, 255, 255])
lower_green = np.array([h - hue_threshold, 80, 80])

r, g, b = (255, 255, 255)
hue_threshold = 5
h = rgb_hue(r, g, b)
upper_line = np.array([h + hue_threshold, 255, 255])
lower_line = np.array([h - hue_threshold, 100, 100])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])

# # define the video path
# VIDEO_FILE = os.path.join("videos", "output1.avi")
# camera = cv2.VideoCapture(VIDEO_FILE)

# Use this block for live video
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
camera.set(cv2.CAP_PROP_FPS, 120)
kernel = np.ones((5,5),np.uint8)

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=640)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    ball_mask = cv2.inRange(hsv, lower_green, upper_green)
    ball_mask = cv2.erode(ball_mask, kernel, iterations=2)
    ball_mask = cv2.dilate(ball_mask, None, iterations=2)

    line_mask = cv2.inRange(hsv, lower_line, upper_line)
    line_mask = cv2.erode(line_mask, kernel, iterations=2)
    line_mask = cv2.dilate(line_mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # find contours in the mask and initialize the current
    # (x, y) center of the line
    line_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # adding the ball mask and line mask
    dst = cv2.add(ball_mask, line_mask)
    combined_mask = cv2.add(ball_mask, line_mask)
    res = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # only proceed if at least one contour was found
    if len(ball_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 3:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        cv2.line(res, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Combined Mask", dst)
    cv2.imshow("original", frame)
    cv2.imshow("res", res)

    key = cv2.waitKey(50) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
