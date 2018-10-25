import imutils
import cv2
import os

in_loc = os.path.join("videos", "NewCamTest6.avi")
cap = cv2.VideoCapture(in_loc)

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FPS, 120)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is not None:
        frame = imutils.resize(frame, width=640)
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
