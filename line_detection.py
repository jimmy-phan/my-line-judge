import time
import cv2
import os
import numpy as np
import imutils


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def rgb_hue(r,g,b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0,0,0]


VIDEO_FILE = os.path.join("videos", "output.avi")

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# r,g,b = (163, 186, 66)
# h = rgb_hue(r, g, b)
# hue_threshold = 5
# # define range of green color in HSV
# upper_green = np.array([h + 5, 255, 255])
# lower_green = np.array([h - 5, 100, 100])


# define the lower and upper boundaries of the line
# in the HSV color space, then initialize the
# list of tracked points
r, g, b = (230, 245, 200)
h = rgb_hue(r, g, b)
hue_threshold = 10
upper_line = np.array([h + hue_threshold, 2500, 2450])
lower_line = np.array([h - hue_threshold, 0, 0])


cap = cv2.VideoCapture(VIDEO_FILE)

while(cap.isOpened()):
    ret, frame = cap.read()

    # frame = imutils.resize(frame, width=640)
    # Convert BGR to HSV
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only green colors
    # mask = cv2.inRange(hsv, lower_green, upper_green)

    # Threshold the HSV image to get only the line
    line_mask = cv2.inRange(hsv, lower_line, upper_line)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= line_mask)
    edges = cv2.Canny(res, 50, 150, apertureSize=3)

    imshape = edges.shape
    lower_left = [imshape[1] / 9, imshape[0]]
    lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(edges, vertices)

    lines = cv2.HoughLines(roi_image, 1, np.pi / 180, 200)
    if not lines is None:
        for r, theta in lines[0]:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('frame', frame)

    # adding the ball mask and line mask
    # dst = cv2.add(mask, line_mask)
    # combined_mask = cv2.add(mask, line_mask)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # these display the different views
    cv2.imshow("Line", line_mask)
    # cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('frame',frame)
    cv2.imshow("roi", roi_image)
    # cv2.imshow("lines", lines)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()