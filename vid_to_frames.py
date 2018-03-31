import time
import cv2
import os
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
fgbg = cv2.createBackgroundSubtractorMOG2()

def rgb_hue(r,g,b):
    # convert the target color to HSV
    target_color = np.uint8([[[b, g, r]]])
    target_color_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    return target_color_hsv[0,0,0]

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed

    r, g, b = (163, 186, 65)
    h = rgb_hue(r, g, b)

    hue_threshold = 5

    # define range of green color in HSV
    upper_green = np.array([h + 5, 255, 255])
    lower_green = np.array([h - 5, 80, 80])

    cap = cv2.VideoCapture(input_loc)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv2.CAP_PROP_FPS, 120)

    # image array to store frames
    im_array = []
    im_array3 = []

    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()

        im_array3.append(frame)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Write the results back to output location.
        # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        im_array.append(mask)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break
    return im_array, im_array3

in_loc = os.path.join("videos", "output.avi")
out_loc = os.path.join("images")

image_array1, image_array2 = video_to_frames(in_loc, out_loc)


im_array2 = []

for idx in range(0,len(image_array1)):
    frame = image_array2[idx]
    greenFrame = image_array1[idx]
    cv2.imshow('frame', frame)
    fgmask = fgbg.apply(greenFrame)
    # different types of filter to clear up noise
    median = cv2.medianBlur(fgmask,35)
    im_array2.append(median)
    cv2.imshow('median Blur', median)
    # bilateral = cv2.bilateralFilter(fgmask, 25, 75, 75)
    # cv2.imshow('bilateral Blur', bilateral)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


newframe = im_array2[0]
for idx in range(0, len(im_array2)):
    # sframe = cv2.addWeighted(newframe, 1, im_array2[idx], 0.225, -0.55556)
    sframe = cv2.addWeighted(newframe, 1, im_array2[idx], .6, -1.45555)
    newframe = sframe
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

kernel = np.ones((51,51),np.uint8)
kernel2 = np.ones((11,11),np.uint8)
kernel3 = np.ones((5,5),np.uint8)
kernel4 = np.ones((20, 1), np.uint8)  # note this is a vertical kernel

closing = cv2.morphologyEx(newframe, cv2.MORPH_CLOSE, kernel)
dilate = cv2.dilate(closing, kernel2)

img = dilate
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)
ret, img = cv2.threshold(img, 50, 50, 0)

element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

done = False
while not done:
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    opening = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel3)
    dilate2 = cv2.dilate(opening, element)
    skel = cv2.bitwise_or(skel, opening)
    img = eroded.copy()
    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True

d_im = cv2.dilate(skel, kernel4, iterations=2)
closing2 = cv2.morphologyEx(d_im, cv2.MORPH_CLOSE, kernel)
e_im = cv2.erode(closing2, kernel2, iterations=1)

cnts = cv2.findContours(e_im.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
c = max(cnts, key=cv2.contourArea)
extBot = tuple(c[c[:, :, 1].argmax()][0])
cv2.circle(e_im, extBot, 8, (255, 255, 0), -1)

while(True):
    cv2.imshow("skel", skel)
    cv2.imshow('closing', closing)
    cv2.imshow('e_im', e_im)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break