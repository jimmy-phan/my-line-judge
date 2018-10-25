# import the necessary packages
import cv2
import numpy as np
import os
import imutils
from collections import deque
import kalman_example3 as kf3
import matplotlib.pyplot as plt
import scipy.interpolate as inter


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
    in_loc = os.path.join("videos", "NewCamTest4.avi")
    cap = cv2.VideoCapture(in_loc)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Create Kalman Filter Object
    kfObj = kf3.create_KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)

    kernel = np.ones((7, 7), np.uint8)
    kernel1 = np.ones((2, 2), np.uint8)
    count = 0
    count2 = 0

    # keep looping
    while cap.isOpened():
        # grab the current frame
        ret, frame = cap.read()
        if frame is not None:
            frame = imutils.resize(frame, width=640)

            ball_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(ball_mask)
            ball_blur = cv2.medianBlur(fgmask, 5)

            ball_cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            # ball_cnts = ball_cnts[0] if imutils.is_cv2() else ball_cnts[1]

            center = None
            if len(ball_cnts) > 0 and ball_cnts is not None:
                c = max(ball_cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                if M["m00"] == 0:
                    M["m00"] = 1
                else:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    count += 1
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)

            count2 = 0
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    count2 +=1
                    # pts.clear()
                    # break
                    continue
                thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)

                # cv2.line(ball_blur, pts[i - 1], pts[i], (255, 255, 0), thickness)
                # cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                cv2.circle(frame, (pts[i].__getitem__(0) , pts[i].__getitem__(1) ), 1, (255, 0, 0), 2, cv2.LINE_AA, 0)
                cv2.circle(ball_blur, (pts[i].__getitem__(0) , pts[i].__getitem__(1) ), 1, (255, 0, 0), 2, cv2.LINE_AA, 0)

                # Used to find all the ball countours on the frame.
                if (i == len(pts)-1) and (count > 10):
                    blur_cnts = cv2.findContours(ball_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    blur_cnts = blur_cnts[0] if imutils.is_cv2() else blur_cnts[1]

                    centers = []
                    for i in range(len(blur_cnts)):
                        c = max(blur_cnts[i], key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        centers.append((x, y))
                    centers.sort()

                    # The list of x and y coordinates of the position of each ball.
                    x1 = np.array([j[0] for j in centers])
                    y1 = np.array([j[1] for j in centers])

                    # s1 = inter.UnivariateSpline(x1, y1, s=0.01 )
                    s1 = inter.InterpolatedUnivariateSpline(x1, y1, k=4)

                    cr_pts = s1.derivative().roots()
                    cr_pts = np.append(cr_pts, (x1[0], x1[-1]))  # also check the endpoints of the interval
                    cr_vals = s1(cr_pts)
                    min_index = np.argmin(cr_vals)
                    max_index = np.argmax(cr_vals)
                    # print("Maximum value {} at {}".format(cr_vals[max_index], cr_pts[max_index]))

                    y_max = cr_vals[max_index]
                    x_max = cr_pts[max_index]
                    print(int(x_max), int(y_max))

                    # x_range = np.linspace(x1.min(), x1.max() + 2)
                    # plt.gca().invert_yaxis()
                    # # Shows the entire path of the x and y coordinates from the array and the spline.
                    # # plt.plot(x1, y1, 'o', x_range, s1(x_range), '-')
                    # # Shows path or the Spline, and the bouncepoint
                    # plt.plot(x_range, s1(x_range), '-', x_max, y_max, 'o')
                    # plt.show()

                    cv2.circle(frame, (int(x_max), int(y_max)), 1, (0, 255, 0), 5, cv2.LINE_AA, 0)
                    cv2.circle(ball_blur, (int(x_max), int(y_max)), 1, (255, 0, 0), 5, cv2.LINE_AA, 0)

                    cv2.imshow("blur final", ball_blur)
                    cv2.imshow("frame final", frame)
                    count = 0
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyWindow("blur final")
                        break

            cv2.imshow("frame", frame)
            # cv2.imshow("fgmask", fgmask)
            cv2.imshow("blur", ball_blur)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
