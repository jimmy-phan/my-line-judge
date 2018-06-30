'''This script will be used to test different line detection
algorithms'''
import cv2
import os
import time
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('line_output.avi', fourcc, 120.0, (640, 480))


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

    cap = cv2.VideoCapture(input_loc)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv2.CAP_PROP_FPS, 120)

    # image array to store frames
    im_array = []

    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=hsv)

        # Write the results back to output location.
        # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        im_array.append(res)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length - 1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds for conversion." % (time_end - time_start))
            break
    return im_array

while True:
    # grab the current frame
    (grabbed, img) = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

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
            out.write(img)
        except NameError:
            pass

        # show the output image
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# in_loc = os.path.join("videos", "line_output.avi")
in_loc = "output.avi"
out_loc = os.path.join("images")

image_array1 = video_to_frames(in_loc, out_loc)

im_array2 = []

for idx in range(0, len(image_array1)):
    frame = image_array1=[idx]
    cv2.imshow('frame', frame)
    # different types of filter to clear up noise
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
