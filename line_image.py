'''This script will be used to test different line detection algorithms'''
# this script finds the lines on the court.
import cv2
import numpy as np
import imutils


def rgb_hue(r, g, b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0, 0, 0]


# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FPS, 120)

# r, g, b = (180, 225, 210)
# hue_threshold = 50
# r, g, b = (176, 206, 199)
# hue_threshold = 50
# h = rgb_hue(r, g, b)
# print(h)
# upper_line = np.array([h + hue_threshold, 45, 225])
# lower_line = np.array([h - hue_threshold, 20, 20])
# #
# r, g, b = (176, 206, 199)
# hue_threshold = 50
# h = rgb_hue(r, g, b)
# print(h)
# upper_line = np.array([h + hue_threshold, 35, 245])
# lower_line = np.array([h - hue_threshold, 5, 0])


while True:
    # grab the current frame
    # (grabbed, img) = cap.read()

    # img = cv2.imread('baseline2.jpg', 0)
    frame = cv2.imread('frame_0.png')
    frame = imutils.resize(frame, width=640)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Fine any object within the range and clear out noise.
    # line_mask = cv2.inRange(hsv, lower_line, upper_line)

    # edges = cv2.Canny(line_mask, 75, 150, apertureSize=3)
    edges = cv2.Canny(frame, 75, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    # print(lines)
    val = None

    # edges = cv2.Canny(frame, 75, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 85, maxLineGap=10)
    # val = None

    if lines is not None:
        max_rho = 0
        theta_threshold = 15
        min_theta = -theta_threshold * np.pi / 180
        max_theta = theta_threshold * np.pi / 180
        for line in lines:
            rho, theta = line[0]
            if (rho > max_rho) and (min_theta <= theta <= max_theta):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)
                max_rho = rho
        try:
            val = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # print(val)
            # cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)
        except NameError:
            pass

    # show the output image
    cv2.imshow("Image", frame)
    cv2.imshow("edges", edges)
    # cv2.imshow("line", line_mask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
