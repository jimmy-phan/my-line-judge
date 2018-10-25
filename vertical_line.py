'''This script will be used to test different line detection algorithms'''
# this script finds the lines on the court.
import cv2
import numpy as np
import imutils


# def rgb_hue(r, g, b):
#     # convert the target color to HSV
#     target_color = np.uint8([[[b, g, r]]])
#     target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
#     return target_color_hsv[0, 0, 0]


# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FPS, 120)

while True:
    # grab the current frame
    # (grabbed, img) = cap.read()

    # frame = cv2.imread('baseline2.jpg', 0)
    frame = cv2.imread('frame_1.png')
    frame = imutils.resize(frame, width=640)

    edges = cv2.Canny(frame, 350, 360, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=6, theta=np.pi/180, threshold=65, maxLineGap=15, minLineLength=120)

    count = 0
    if lines is not None:
        # Each line is an array of arrays, we use line[0] to get the first element of the array
        for line in lines:
            x1, y1, x2, y2 = line[0]
            count += 1
            if count == 8:
                line = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # line = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.putText(frame, "{}".format(count), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

    print(line)

    # show the output image
    cv2.imshow("Image", frame)
    cv2.imshow("edges", edges)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()
