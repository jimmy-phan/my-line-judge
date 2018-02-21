import time
import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
fgbg = cv2.createBackgroundSubtractorMOG2()

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
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        im_array.append(frame)
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
    return im_array

in_loc = os.path.join("videos", "output.avi")
out_loc = os.path.join("images")
image_array = video_to_frames(in_loc, out_loc)


im_array2 = []

for idx in range(0,len(image_array)):
    frame = image_array[idx]
    cv2.imshow('frame', frame)
    fgmask = fgbg.apply(frame)
    im_array2.append(fgmask)
    cv2.imshow('avg',fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
sframe = im_array2[0]

im_array3 = []
for idx in range(0,len(im_array2)):
    # sframe = sum(im_array2)
    sframe = cv2.addWeighted(sframe, .25, im_array2[idx], .75, 0)
    # sframe = cv2.add(sframe, im_array2[idx])
    newframe = sframe
    cv2.imshow('sframe', newframe)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
