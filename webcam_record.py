import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('NewCamTest6.avi', fourcc, (cap.get(cv2.CAP_PROP_FPS)), (640, 480))

while True:
    ret, frame = cap.read()

    k = cv2.waitKey(1)
    if k % 256 == 32:
        print("Recording...")
        while True:
            ret, frame = cap.read()
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped...")
                break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

    # k = cv2.waitKey(1)
    # if k % 256 == 32:
    #     print("Recording...")
    #     if ret:
    #         qqqq# frame = cv2.flip(frame, 0)
    #
    #         # # write the flipped frame
    #         out.write(frame)
    #
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("q pressed Exit camera")
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     print("Camera closed")
    #     break

    # if ret:
    #     # frame = cv2.flip(frame, 0)
    #
    #     # # write the flipped frame
    #     out.write(frame)
    #
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # else:
    #     break


# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()