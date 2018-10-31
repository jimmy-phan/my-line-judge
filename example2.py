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

    # r, g, b = (232, 255, 193)
    # hue_threshold = 20
    # h = rgb_hue(r, g, b)
    # upper_green = np.array([h + hue_threshold, 230, 230])
    # lower_green = np.array([h - hue_threshold, 75, 75])

    # r, g, b = (232, 255, 193)
    # hue_threshold = 10
    # h = rgb_hue(r, g, b)
    # upper_green = np.array([h + hue_threshold, 230, 230])
    # lower_green = np.array([h - hue_threshold, 55, 55])

    pts = deque(maxlen=64)
    temp1 = deque(maxlen=64)

    # in_loc = os.path.join("videos", "test2.mp4")
    in_loc = os.path.join("videos", "NewCamTest5.avi")
    cap = cv2.VideoCapture(in_loc)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Create Kalman Filter Object
    kfObj = kf3.create_KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)

    kernel = np.ones((7, 7), np.uint8)
    kernel1 = np.ones((2, 2), np.uint8)

    # keep looping
    while cap.isOpened():
        # grab the current frame
        ret, frame = cap.read()
        if frame is not None:
            frame = imutils.resize(frame, width=640)


            ball_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(ball_mask)
            ball_blur = cv2.medianBlur(fgmask, 5)

            circles = cv2.HoughCircles(image=ball_blur,
                                       method=cv2.HOUGH_GRADIENT,
                                       dp=1, minDist=50,  # minDist=h / 10,
                                       # type, 1/scale, min center dists
                                       param1=200, param2=9,  # params1?, param2?
                                       minRadius=2, maxRadius=16)  # min radius, max radius

            # if circles is not None and len(circles) > 0:
            #     cc = circles[0].tolist()
            #     cc = [cc[0]]
            #
            #     temp = True
            #     for (x, y, radius) in cc:
            #         x = int(x)
            #         y = int(y)
            #         radius = int(radius)
            #
            #         if temp:
            #             # cv2.circle(frame, (x, y), radius, (255, 0, 0), 2, cv2.LINE_AA, 0)
            #             temp = False
            #         else:
            #             # cv2.circle(frame, (x, y), radius, (0, 0, 255), 2, cv2.LINE_AA, 0)

            ball_cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ball_cnts = ball_cnts[0] if imutils.is_cv2() else ball_cnts[1]
            center = None

            if len(ball_cnts) > 0:
                c = max(ball_cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                M = cv2.moments(c)
                if M["m00"] == 0:
                    M["m00"] = 1
                else:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

            pts.appendleft(center)

            line = None
            count = 0
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    # pts.pop()
                    count += 1
                    # print(count)
                    # break
                # if count == 1:
                #     count = 0
                #     # break
                #     # print(i)
                    continue
                thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
                cv2.line(ball_blur, pts[i - 1], pts[i], (255, 255, 0), thickness)
                line = cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                if i == len(pts)-1:
                    res = cv2.bitwise_and(frame, frame, mask=ball_mask)
                    cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                    c = max(cnts, key=cv2.contourArea)
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    circle = cv2.circle(ball_blur, extBot, 6, (255, 255, 0), -1)
                    cv2.imshow("blur final", ball_blur)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        continue
                        # break

            cv2.imshow("frame", frame)
            cv2.imshow("fgmask", fgmask)
            cv2.imshow("blur", ball_blur)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
