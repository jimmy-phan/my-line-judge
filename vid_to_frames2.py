import time
import cv2
import os
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

# def video_to_frames(input_loc, output_loc):
#     """Function to extract frames from input video file
#     and save them as separate frames in an output directory.
#     Args:
#         input_loc: Input video file.
#         output_loc: Output directory to save the frames.
#     Returns:
#         None
#     """
#     try:
#         os.mkdir(output_loc)
#     except OSError:
#         pass
#     # Log the time
#     time_start = time.time()
#     # Start capturing the feed
#
#     cap = cv2.VideoCapture(input_loc)
#     # cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
#     cap.set(cv2.CAP_PROP_FPS, 120)
#
#     # image array to store frames
#     im_array = []
#
#     # Find the number of frames
#     # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
#     # print ("Number of frames: ", video_length)
#     # count = 0
#     print ("Converting video..\n")
#     # Start converting the video
#     while cap.isOpened():
#         # Extract the frame
#         ret, frame = cap.read()
#
#         # Convert BGR to HSV
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#         # Bitwise-AND mask and original image
#         res = cv2.bitwise_and(frame, frame, mask=hsv)
#
#         # Write the results back to output location.
#         # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
#         im_array.append(res)
#         # count = count + 1
#         # # If there are no more frames left
#         # if (count > (video_length-1)):
#         #     # Log the time again
#         #     time_end = time.time()
#         #     # Release the feed
#         #     cap.release()
#         #     # Print stats
#         #     print ("Done extracting frames.\n%d frames extracted" % count)
#         #     print ("It took %d seconds for conversion." % (time_end-time_start))
#         #     break
#     return im_array
#
# # in_loc = os.path.join("videos", "output.avi")
# in_loc = 0
# out_loc = os.path.join("images")
#
# image_array1 = video_to_frames(in_loc, out_loc)
#
#
# im_array2 = []
#
# for idx in range(0,len(image_array1)):
#     frame = image_array1[idx]
#     cv2.imshow('frame', frame)
#     # different types of filter to clear up noise
#     # bilateral = cv2.bilateralFilter(fgmask, 25, 75, 75)
#     # cv2.imshow('bilateral Blur', bilateral)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


im_array = []

while cap.isOpened():
    # Extract the frame
    ret, frame = cap.read()

    # # Convert BGR to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=hsv)

    # Write the results back to output location.
    # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
    im_array.append(frame)
    cv2.imshow('res', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



for idx in range(0,len(im_array)):
    frame = im_array[idx]

    # # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # # Bitwise-AND mask and original image

    cv2.imshow('frame', frame)
    cv2.imshow('hsv', hsv)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
