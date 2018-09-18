import cv2
import time
import imutils
import argparse
import os

if __name__ == '__main__':

    in_loc = os.path.join("videos", "test8.avi")
    cap = cv2.VideoCapture(in_loc)


    # Start default camera
    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv2.CAP_PROP_FPS, 120)

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=100, help="# of frames to loop over for FPS test")
    ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
    args = vars(ap.parse_args())

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    # Number of frames to capture
    num_frames = 240

    print("Capturing {0} frames".format(num_frames))
    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = cap.read()
        #
        # # check to see if the frame should be displayed to our screen
        # if frame is not None:
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     cv2.imshow("Frame", gray)
        #     key = cv2.waitKey(1) & 0xFF
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    # Release video
    cap.release()

# import cv2
# import os
#
#
# in_loc = os.path.join("videos", "test8.avi")
# cap = cv2.VideoCapture(in_loc)
#
# numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#
# # fps = cap.get(cv2.CAP_PROP_FPS)
# #
# # print(fps)
#
# print(numFrames)
#
