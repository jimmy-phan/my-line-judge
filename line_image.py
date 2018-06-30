'''This script will be used to test different line detection
algorithms'''
# this script finds the lines on the court.
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

while True:
    # grab the current frame
    # (grabbed, img) = cap.read()

    img = cv2.imread('baseline.jpg', 0)

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    if lines is not None:
        max_rho = 0
        theta_threshold = 45
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
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                max_rho = rho
        try:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        except NameError:
            pass

    # show the output image
    cv2.imshow("Image", img)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
