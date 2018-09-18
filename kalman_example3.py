import cv2
import numpy as np
import os
import imutils
from collections import deque
import vid_to_frames as vf


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


class create_KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    # kf.measurementMatrix = np.array([[1, 0, 0, 0],
    #                                  [0, 1, 0, 0]], np.float32)
    # kf.transitionMatrix = np.array([[1, 0, 1, 0],
    #                                 [0, 1, 0, 1],
    #                                 [0, 0, 1, 0],
    #                                 [0, 0, 0, 1]], np.float32)

    # The processNoiseCov matrix should mirror the transition matrix size
    kf.measurementMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    # kf.processNoiseCov = np.array([[1, 0, 0, 0],
    #                                [0, 1, 0, 0],
    #                                [0, 0, 1, 0],
    #                                [0, 0, 0, 1]], np.float32) * .05
    # kf.measurementNoiseCov = 1e-1 * np.ones((2, 2))
    #
    # kf.measurementNoiseCov = np.array([[1, 1],
    #                                    [1, 1]], np.float32) * 1e-1

    # kf.errorCovPost = np.array([[1, 1, 1, 1],
    #                             [1, 1, 1, 1],
    #                             [1, 1, 1, 1],
    #                             [1, 1, 1, 1]], np.float32)*1.
    #
    kf.statePost = 0.1 * np.random.randn(4, 2)

    def estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


if __name__ == '__main__':
    actualArr = deque(maxlen=64)
    predArr = deque(maxlen=64)

    # Color Codes used to draw the lines on the screen.
    RED = (0, 0, 255)   # Actual Path
    BLUE = (255, 0, 0)  # Predicted Path
    YELLOW = (0, 255, 0)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Create Kalman Filter Object
    kfObj = create_KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)

    in_loc = os.path.join("videos", "test2.mp4")
    # in_loc = os.path.join("videos", "output1.avi")
    # in_loc = os.path.join("videos", "test3.avi")
    cap = cv2.VideoCapture(in_loc)

    # generate the green mask for the ball
    # r, g, b = (163, 186, 65)
    # r, g, b = (126, 128, 58)
    # r, g, b = (220, 255, 30)
    # r, g, b = (206, 235, 130)
    r, g, b = (115, 255, 95)
    # r, g, b = (120, 130, 60)

    hue_threshold = 25
    h = rgb_hue(r, g, b)
    # upper_green = np.array([h + hue_threshold, 255, 255])
    # lower_green = np.array([h - hue_threshold, 80, 80])

    upper_green = np.array([h + hue_threshold, 250, 250])
    lower_green = np.array([h - hue_threshold, 45, 45])

    # create the structuring elements
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((9, 9), np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame = imutils.resize(frame, width=640)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Fine any object within the range and clear out noise.
            ball_mask = cv2.inRange(hsv, lower_green, upper_green)
            ball_mask = cv2.erode(ball_mask, kernel, iterations=4)
            ball_mask = cv2.dilate(ball_mask, kernel, iterations=4)
            # Apply a background subtractor to only focus on moving objects.
            fgmask = fgbg.apply(ball_mask)
            ball_blur = cv2.medianBlur(fgmask, 35)
            ball_blur = cv2.erode(ball_blur, kernel, iterations=2)
            ball_blur = cv2.dilate(ball_blur, kernel)
            ball_blur = cv2.morphologyEx(ball_blur, cv2.MORPH_CLOSE, kernel)
            # frame = ball_blur
            # cv2.imshow('Ball Mask', ball_mask)
            cv2.imshow('FGMASK', fgmask)
            cv2.imshow('Ball Blur', ball_blur)

            # find contours in the mask and initialize the current (x, y) center of the ball
            # ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            ball_cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            # Find ball blob as it is the biggest green object in the frame
            # [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(ball_mask, 8, cv2.CV_32S)
            [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(ball_blur, 8, cv2.CV_32S)

            # First biggest contour is image border always, Remove it
            stats = np.delete(stats, 0, axis=0)

            # if len(ball_cnts) > 0:
            if len(stats) > 0:
                maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)
                # This is our ball coordinatess that needs to be tracked
                ballX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2] / 2)
                ballY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3] / 2)

                # Determines the predicted coordinates using the KalmanFilter.
                predictedCoords = kfObj.estimate(ballX, ballY)

                # Draw Actual coordinates from segmentation
                cv2.circle(frame, (int(ballX), int(ballY)), 20, [0, 0, 255], 2, 8)
                cv2.line(frame, (int(ballX), int(ballY) + 20), (int(ballX) + 50, int(ballY) + 20), [100, 100, 255], 2, 8)
                cv2.putText(frame, "Actual", (int(ballX) + 50, int(ballY) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
                Actual = (int(ballX), int(ballY))

                # Draw Kalman Filter Predicted output
                # cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
                # cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                #          (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
                # cv2.putText(frame, "Predicted", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, [50, 200, 250])
                Predicted = (predictedCoords[0], predictedCoords[1])

                print(Predicted)
            # update the points arrays for Actual and predicted
            actualArr.appendleft(Actual)
            predArr.appendleft(Predicted)

            # TODO : Currently the project only works from right to left need to fix that
            # Draw lines on the frame connecting all the points.
            for i in range(1, len(actualArr)):
                # Check if the value in pts[i-1] is less than pts[i] that means
                # the ball should always be coming from right to left
                # pts[i].__getitem__(0) means the first item in the tuple at the given index
                if actualArr[i - 1].__getitem__(0) < actualArr[i].__getitem__(0):
                    # print((actualArr[i - 1].__getitem__(0), actualArr[i].__getitem__(0) ))
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    cv2.line(frame, actualArr[i - 1], actualArr[i], RED, thickness)
                # if counter >= 10 and i == 1 and pts[-10] is not None:
                #     dX = pts[-10][0] - pts[i][0]
                #     dY = pts[-10][1] - pts[i][1]
                #     # (dirX, dirY)) = ("", "")
                #     print(dX, dY)
                    # if dY > 0:
                    #     print("bounce", (dX, dY))
            # Duplicated from above but for the predicted path.
            for i in range(1, len(predArr)):
                if predArr[i - 1].__getitem__(0) < predArr[i].__getitem__(0):
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    cv2.line(frame, predArr[i - 1], predArr[i], BLUE, thickness)

            # for i in range(2, len(predArr)):
            #     if predArr[i-1] is None or predArr[i] is None:
            #         continue
            #     thickness = int(np.sqrt(args["buffer"] / float(i+1))*1.5)
            #     cv2.line(frame, predArr[i-1], predArr[i], (0, 0, 255), thickness)

            # for i in range(1, len(actualArr)):
            #     # if either of the tracked points are None, ignore them
            #     if actualArr[i - 1] is None or actualArr[i] is None:
            #         continue
            #     # otherwise, compute the thickness of the line and
            #     # draw the connecting lines
            #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
            #     cv2.line(frame, actualArr[i - 1], actualArr[i], (0, 0, 255), thickness)

            cv2.imshow('Input', frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
