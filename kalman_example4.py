import cv2
import numpy as np
import os
import imutils
from collections import deque
import argparse


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


def draw_traceline(ball_cnts, frame):
    # only proceed if at least one contour was found
    # find the largest contour in the mask,
    # then use it to compute the minimum enclosing circle and centroid
    center = None
    if len(ball_cnts) > 0:
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        # was originally set to 3, change this number to vary the radius.
        if radius > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # update the points queue
    pts.appendleft(center)
    return pts


def draw_predicted(stats, frame):
    maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)
    # This is our ball coords that needs to be tracked
    ballX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2] / 2)
    ballY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3] / 2)

    predictedCoords = kfObj.Estimate(ballX, ballY)
    # Draw Actual coords from segmentation
    cv2.circle(frame, (int(ballX), int(ballY)), 20, [0, 0, 255], 2, 8)
    cv2.line(frame, (int(ballX), int(ballY) + 20), (int(ballX) + 50, int(ballY) + 20), [100, 100, 255], 2, 8)
    cv2.putText(frame, "Actual", (int(ballX) + 50, int(ballY) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

    # Draw Kalman Filter Predicted output
    cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
    cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
             (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
    Predicted = (predictedCoords[0],predictedCoords[1])
    cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, [50, 200, 250])
    pts.appendleft(Predicted)


def draw_path(pts):
    for i in range(1, len(pts)):
        if pts[i - 1].__getitem__(0) < pts[i].__getitem__(0):
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
            cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)
    return frame


class create_KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])
pts2 = deque(maxlen=args["buffer"])
pts3 = deque(maxlen=args["buffer"])

# Color Codes
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 0)

fgbg = cv2.createBackgroundSubtractorMOG2()

# Create Kalman Filter Object
kfObj = create_KalmanFilter()
predictedCoords = np.zeros((2, 1), np.float32)

# in_loc = os.path.join("videos", "test2.mp4")
# in_loc = os.path.join("videos", "output1.avi")
in_loc = os.path.join("videos", "test1.avi")
cap = cv2.VideoCapture(in_loc)

# generate the green mask for the ball
# r, g, b = (163, 186, 65)
# r, g, b = (220, 255, 30)
# r, g, b = (206, 235, 130)
r, g, b = (115, 255, 95)

hue_threshold = 30
h = rgb_hue(r, g, b)
# upper_green = np.array([h + hue_threshold, 255, 255])
# lower_green = np.array([h - hue_threshold, 80, 80])

upper_green = np.array([h + hue_threshold, 250, 250])
lower_green = np.array([h - hue_threshold, 45, 45])

# create the structuring elements
kernel = np.ones((5,5),np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

while cap.isOpened():
    ret, frame = cap.read()
    if frame is not None:
        frame = imutils.resize(frame, width=640)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_mask = cv2.inRange(hsv, lower_green, upper_green)
        ball_mask = cv2.erode(ball_mask, kernel)
        ball_mask = cv2.dilate(ball_mask, kernel, iterations=4)
        ball_mask = fgbg.apply(ball_mask)
        ball_mask = cv2.medianBlur(ball_mask, 35)
        # Erode the frame to clear up any noise
        greenMaskEroded = cv2.erode(ball_mask, kernel)
        greenMaskEroded = cv2.dilate(greenMaskEroded, kernel, iterations=4)
        cv2.imshow('Thresholded', ball_mask)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # Find ball blob as it is the biggest green object in the frame
        [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(ball_mask, 8, cv2.CV_32S)

        # First biggest contour is image border always, Remove it
        stats = np.delete(stats, 0, axis=0)

        if len(ball_cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            # c = max(ball_cnts, key=cv2.contourArea)
            # ((x, y), radius) = cv2.minEnclosingCircle(c)
            # M = cv2.moments(c)
            # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #
            # # only proceed if the radius meets a minimum size
            # # was originally set to 3, change this number to vary the radius.
            # if radius > 2:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(frame, (int(x), int(y)), int(radius),
            #                (0, 255, 255), 2)
            #     cv2.circle(frame, center, 5, (0, 0, 255), -1)

            maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)
            # This is our ball coords that needs to be tracked
            ballX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2] / 2)
            ballY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3] / 2)

            predictedCoords = kfObj.Estimate(ballX, ballY)

            # Draw Actual coords from segmentation
            cv2.circle(frame, (int(ballX), int(ballY)), 20, [0, 0, 255], 2, 8)
            cv2.line(frame, (int(ballX), int(ballY) + 20), (int(ballX) + 50, int(ballY) + 20), [100, 100, 255], 2, 8)
            cv2.putText(frame, "Actual", (int(ballX) + 50, int(ballY) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
            Actual = (int(ballX), int(ballY))

            # Draw Kalman Filter Predicted output
            cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
            cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                     (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
            cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, [50, 200, 250])
            Predicted = (predictedCoords[0], predictedCoords[1])
        # update the points queue
        pts.appendleft(Actual)
        pts2.appendleft(Predicted)
        pts3.appendleft(center)

        # Draw lines on the frame connecting all the points.
        for i in range(1, len(pts)):
            if pts[i - 1].__getitem__(0) < pts[i].__getitem__(0):
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                cv2.line(frame, pts[i - 1], pts[i], RED, thickness)
        for i in range(1, len(pts2)):
            if pts2[i - 1].__getitem__(0) < pts2[i].__getitem__(0):
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                cv2.line(frame, pts2[i - 1], pts2[i], BLUE, thickness)
        # for i in range(1, len(pts3)):
        #     # if either of the tracked points are None, ignore them
        #     if pts3[i - 1] is None or pts3[i] is None:
        #         continue
        #     # otherwise, compute the thickness of the line and
        #     # draw the connecting lines
        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
        #     cv2.line(frame, pts3[i - 1], pts3[i], YELLOW, thickness)

        cv2.imshow('Input', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
