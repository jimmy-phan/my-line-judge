# import the necessary packages
import numpy as np
import imutils
import cv2
import os


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        # This function is used to estimate the position of the Object
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


# establish the color thresholds and convert from rgb to hsv for green
r, g, b = (115, 255, 95)
hue_threshold = 30
h = rgb_hue(r, g, b)
upper_green = np.array([h + hue_threshold, 250, 250])
lower_green = np.array([h - hue_threshold, 45, 45])

# # define the video path
VIDEO_FILE = os.path.join("videos", "balltest.avi")
cap = cv2.VideoCapture(VIDEO_FILE)

# Use this block for live video
# camera = cv2.VideoCapture(1)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# camera.set(cv2.CAP_PROP_FPS, 120)

kernel = np.ones((3,3),np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

KalmanFilterObj = KalmanFilter()
predictedCoords = np.zeros((2, 1), np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if frame is not None:
        frame = imutils.resize(frame, width=640)

        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        ball_mask = cv2.inRange(hsv, lower_green, upper_green)
        ball_mask = cv2.erode(ball_mask, kernel, iterations=4)
        ball_mask = cv2.dilate(ball_mask, element, iterations=4)

        # Find ball blob as it is the biggest green object in the frame
        [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(ball_mask, 8, cv2.CV_32S)

        # First biggest contour is image border always, Remove it
        stats = np.delete(stats, 0, axis=0)
        maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)

        # This is our ball coords that needs to be tracked
        ballX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2] / 2)
        ballY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3] / 2)

        predictedCoords = KalmanFilterObj.Estimate(ballX, ballY)

        # Draw Actual coords from segmentation
        cv2.circle(frame, (ballX, ballY), 20, [0, 0, 255], 2, 8)
        cv2.line(frame, (ballX, ballY + 20), (ballX + 50, ballY + 20), [100, 100, 255], 2, 8)
        cv2.putText(frame, "Actual", (ballX + 50, ballY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        # Draw Kalman Filter Predicted output
        cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
        cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
        cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, [50, 200, 250])
        cv2.imshow('Input', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()