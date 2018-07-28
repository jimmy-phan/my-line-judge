import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

count = 0
arr = []

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE pressed
        img = "frame_{}.png".format(count)
        cv2.imwrite(img, frame)
        arr.append(frame)
        print("{} written!".format(img))
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for idx in range(0,len(arr)):
    frame = arr[idx]
    cv2.imshow("Arr Pics", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
