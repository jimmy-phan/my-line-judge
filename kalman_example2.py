import cv2
import numpy as np
import os
import imutils


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


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

# ball_cnts = null


# Create Kalman Filter Object
kfObj = create_KalmanFilter()
predictedCoords = np.zeros((2, 1), np.float32)

in_loc = os.path.join("videos", "test2.mp4")
# in_loc = os.path.join("videos", "output1.avi")
cap = cv2.VideoCapture(in_loc)

# generate the green mask for the ball
# r, g, b = (163, 186, 65)
r, g, b = (214, 255, 17)
hue_threshold = 5
h = rgb_hue(r, g, b)
upper_green = np.array([h + hue_threshold, 255, 255])
lower_green = np.array([h - hue_threshold, 80, 80])

# create the structuring elements
kernel = np.ones((5,5),np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

while cap.isOpened():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    if frame is not None:
        ball_mask = cv2.inRange(frame, lower_green, upper_green)
        ball_mask = cv2.erode(ball_mask, kernel)
        ball_mask = cv2.dilate(ball_mask, kernel, iterations=4)

        # res = cv2.bitwise_and(frame, frame, mask=greenMask,)
        # gray = cv2.cvtColor(greenMask, cv2.COLOR)
        # fgmask = fgbg.apply(greenMask)

        # Erode the frame to clear up any noise
        greenMaskEroded = cv2.erode(ball_mask, kernel)
        greenMaskEroded = cv2.dilate(greenMaskEroded, kernel, iterations=4)

        # greenMaskDilated = cv2.dilate(greenMaskEroded, kernel)
        cv2.imshow('Thresholded', ball_mask)
        # cv2.imshow('Thresholded', greenMaskDilated)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # cv2.imshow('fgmask', greenMask)
        # Find ball blob as it is the biggest green object in the frame
        # [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(greenMaskDilated, 8, cv2.CV_32S)
        [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(ball_mask, 8, cv2.CV_32S)

        # First biggest contour is image border always, Remove it
        stats = np.delete(stats, (0), axis=0)
        # print("the stats string", stats)

        # only proceed if at least one contour was found

        # if len(ball_cnts) > 0:
        if len(stats) > 0:
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
            cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, [50, 200, 250])
            cv2.imshow('Input', frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
cv2.destroyAllWindows()
