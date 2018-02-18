import cv2
import time
import imutils
# The following imports are used for the multi-threading
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
import argparse

if __name__ == '__main__':

    # Start default camera
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    video.set(cv2.CAP_PROP_FPS, 120)

    # The next block is for the multi-threading
    # video = WebcamVideoStream(src=0).start()
    # fps = FPS().start()

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=100,
                    help="# of frames to loop over for FPS test")
    ap.add_argument("-d", "--display", type=int, default=-1,
                    help="Whether or not frames should be displayed")
    args = vars(ap.parse_args())

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    # if int(major_ver) < 3:
    #     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #     print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    # else:
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Number of frames to capture
    num_frames = 120

    print("Capturing {0} frames".format(num_frames))
    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = video.read()

        # Block is used to grab frames for the multi-threading block
        # frame = video.read()
        # frame = imutils.resize(frame, width=400)

        # check to see if the frame should be displayed to our screen
        # if args["display"] > 0:
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     cv2.imshow("Frame", gray)
        #     key = cv2.waitKey(1) & 0xFF

        # update the FPS counter
        # used for the multi-threading
        # fps.update()

    # End time

    # fps.stop()       #Used for the multi-threading
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    # Release video
    video.release()
    # video.stop()    # used for the multi-threading
