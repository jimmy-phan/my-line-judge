import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

video = WebcamVideoStream(src=0).start()
fps = FPS().start()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 120.0, (640, 480))

# while(video.isOpened()):
while (True):
    f = video.read()
    if frame is not None:
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
# video.release()
video.stop()
out.release()
cv2.destroyAllWindows()
