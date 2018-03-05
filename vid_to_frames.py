import time
import cv2
import os
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
# fgbg = cv2.createBackgroundSubtractor()
fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.createBackgroundSubtractorKNN()

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
    # im_array2.append(fgmask)
    # cv2.imshow('avg',fgmask)
    # cv2.imshow('res', res)
    # different types of filter to clear up noise
    median = cv2.medianBlur(fgmask,35)
    im_array2.append(median)
    cv2.imshow('median Blur', median)
    # bilateral = cv2.bilateralFilter(fgmask, 25, 75, 75)
    # cv2.imshow('bilateral Blur', bilateral)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# setframe = im_array2[82]
# setframe2 = im_array2[83]
# # cv2.imshow('setframe', setframe)
# # cv2.imshow('setframe2', setframe2)
# displayVal = cv2.addWeighted(setframe, 1, setframe2, 1, 0)
# display1 = cv2.addWeighted(displayVal, 1, im_array2[84], 1, 0)
# display2 = cv2.addWeighted(display1, 1, im_array2[85], 1, 0)
# display3 = cv2.addWeighted(display2, 1, im_array2[86], 1, 0)
# display4 = cv2.addWeighted(display3, 1, im_array2[87], 1, 0)
# display5 = cv2.addWeighted(display4, 1, im_array2[88], 1, 0)
# display6 = cv2.addWeighted(display5, 1, im_array2[89], 1, 0)
# display7 = cv2.addWeighted(display6, 1, im_array2[90], 1, 0)
# display8 = cv2.addWeighted(display7, 1, im_array2[1], 1/8, 0)
# display9 = cv2.addWeighted(display8, 1, im_array2[2], 1/9, 0)
# display10 = cv2.addWeighted(display9, 1, im_array2[3], 1/10, 0)
# display11 = cv2.addWeighted(display10, 1, im_array2[4], 1, 0)
#
# cv2.imshow('display7', display7)
# cv2.imshow('display11', display10)
newframe = im_array2[0]
for idx in range(0, len(im_array2)):
    # sframe = cv2.addWeighted(newframe, 1, im_array2[idx], 0.225, -0.55556)
    sframe = cv2.addWeighted(newframe, 1, im_array2[idx], .6, -1.45555)
    # sframe = cv2.add(sframe, im_array2[idx])q
    newframe = sframe
    cv2.imshow('sframe', newframe)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

while(True):
    # cv2.imshow('sframe', newframe)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break