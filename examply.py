# import the necessary packages
import cv2
import numpy as np
import os
import imutils
from collections import deque
import kalman_example3 as kf3


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


if __name__ == '__main__':
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the list of tracked points

    # r, g, b = (115, 255, 95)
    # hue_threshold = 25
    # h = rgb_hue(r, g, b)
    # upper_green = np.array([h + hue_threshold, 250, 250])
    # lower_green = np.array([h - hue_threshold, 45, 45])

    r, g, b = (202, 219, 161)
    hue_threshold = 5
    h = rgb_hue(r, g, b)
    upper_green = np.array([h + hue_threshold, 232, 232])
    lower_green = np.array([h - hue_threshold, 95, 95])

    # r, g, b = (232, 229, 193)
    # hue_threshold = 5
    # h = rgb_hue(r, g, b)
    # upper_green = np.array([h + hue_threshold, 230, 230])
    # lower_green = np.array([h - hue_threshold, 78, 78])

    pts = deque(maxlen=64)
    predArr = deque(maxlen=64)
    temp = deque(maxlen=64)

    # in_loc = os.path.join("videos", "test2.mp4")
    in_loc = os.path.join("videos", "NewCamTest5.avi")
    cap = cv2.VideoCapture(in_loc)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Create Kalman Filter Object
    kfObj = kf3.create_KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)

    kernel = np.ones((3, 3), np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # keep looping
    while cap.isOpened():
        # grab the current frame
        ret, frame = cap.read()
        if frame is not None:
            frame = imutils.resize(frame, width=640)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            ball_mask = cv2.inRange(hsv, lower_green, upper_green)
            # ball_mask = cv2.erode(ball_mask, kernel, iterations=2)
            ball_mask = cv2.dilate(ball_mask, element, iterations=4)

            fgmask = fgbg.apply(ball_mask)
            ball_blur = cv2.medianBlur(fgmask, 5)
            # ball_blur = cv2.erode(ball_blur, kernel, iterations=2)
            ball_blur = cv2.dilate(ball_blur, kernel)
            ball_blur = cv2.morphologyEx(ball_blur, cv2.MORPH_CLOSE, kernel, iterations=5)

            # find contours in the mask and initialize the current (x, y) center of the ball
            ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            Predicted = None

            if len(ball_cnts) > 0:
                c = max(ball_cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                predictedCoords = kfObj.estimate(x, y)
                Predicted = (predictedCoords[0], predictedCoords[1])

            # update the points queue
            pts.appendleft(center)
            predArr.appendleft(Predicted)

            cicle = None
            line = None
            count = 0
            predLine = None
            predCount = 0

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                    # pts.pop()
                    # break
                thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
                cv2.line(ball_blur, pts[i - 1], pts[i], (255, 255, 0), thickness)
                line = cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                count += 1
            #
            #     if line is not None and count > 2:
            #         res = cv2.bitwise_and(frame, frame, mask=ball_mask)
            #         cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #         cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            #         c = max(cnts, key=cv2.contourArea)
            #         extBot = tuple(c[c[:, :, 1].argmax()][0])
            #         cv2.circle(ball_blur, extBot, 8, (255, 255, 0), -1)
            #
            #         cv2.imshow("blur frame", ball_blur)
            #         if cv2.waitKey(0) & 0xFF == ord('q'):
            #             break
            #         continue

            for i in range(1, len(predArr)):
                if predArr[i - 1] is None or predArr[i] is None:
                    continue
                # print(predArr[i])
                thickness = int(np.sqrt(64/ float(i + 1)) * 1.5)
                line = cv2.line(frame, predArr[i - 1], predArr[i], (0, 255, 0), thickness)

            # for i in range(1, len(predArr)):
            #     # if map(itemgetter(0), predArr[i-1]) > map(itemgetter(0), predArr[i]):
            #     if predArr[i-1] is None or predArr[i] is None:
            #         predArr.pop()
            #         break
            #     if predArr[i - 1].__getitem__(0) < predArr[i].__getitem__(0):
            #         thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
            #         predLine = cv2.line(frame, predArr[i - 1], predArr[i], (255, 0, 0), thickness)
            #         cv2.line(ball_blur, predArr[i - 1], predArr[i], (255, 255, 0), thickness)
            #
            #         predCount += 1

            # if (predLine is not None and predCount > 2) or (line is not None and count > 2):
            #     # if (predLine is not None and predCount > 2):
            #     res = cv2.bitwise_and(frame, frame, mask=ball_mask)
            #     cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            #     c = max(cnts, key=cv2.contourArea)
            #     extBot = tuple(c[c[:, :, 1].argmax()][0])
            #     circle = cv2.circle(ball_blur, extBot, 6, (255, 255, 0), -1)
            #
            #     print("this coords: ", extBot)
            #     cv2.imshow("blur frame", ball_blur)
            #     if cv2.waitKey(0) & 0xFF == ord('q'):
            #         # break
            #         continue

                # cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                # c = max(cnts, key=cv2.contourArea)
                # extBot = tuple(c[c[:, :, 1].argmax()][0])
                # cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

            cv2.imshow("frame", frame)
            cv2.imshow("blur", ball_blur)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
