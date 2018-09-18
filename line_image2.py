import cv2
import numpy as np
import vid_to_frames as vf
import os


# img = cv2.imread('baseline.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),11)
#
#     cv2.imshow("Image", img)
#     cv2.imshow("edges", edges)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
#
#
# cv2.destroyAllWindows()

fgbg = cv2.createBackgroundSubtractorMOG2()

in_loc = os.path.join("videos", "output1.avi")


image_array1, image_array2 = vf.video_to_frames(in_loc)

im_array3 = []

for idx in range(0,len(image_array2)):
    frame = image_array1[idx]
    greenFrame = image_array2[idx]
    cv2.imshow('frame', frame)
    fgmask = fgbg.apply(greenFrame)
    # different types of filter to clear up noise
    median = cv2.medianBlur(fgmask,35)
    im_array3.append(median)
    cv2.imshow('median Blur', median)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()