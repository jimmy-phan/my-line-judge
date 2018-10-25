'''This script will be used to test different line detection algorithms'''
# this script finds the lines on the court.
import cv2
import numpy as np
import imutils


def find_line(frame):

    while True:
        # in_loc = os.path.join("images", "frame_1.png")
        # frame = cv2.imread(in_loc)
        frame = imutils.resize(frame, width=640)

        edges = cv2.Canny(frame, 350, 350, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi/180, threshold=74, maxLineGap=9, minLineLength=130)

        count = 0
        if lines is not None:
            # Each line is an array of arrays, we use line[0] to get the first element of the array
            for line in lines:
                x1, y1, x2, y2 = line[0]
                count += 1
                # if count == 3:
                #     line = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     line2 = [(x1, y1), (x2, y2)]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "{}".format(count), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
        # print(lines[2])

        cv2.imshow("Line", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyWindow("Line")
            var = input("Which line do you want: ")
            return lines[int(var)-1]


# if __name__ == '__main__':
    #
    # frame = cv2.imread('frame_1.png')
    # lines = find_line(frame)
    # print(lines)
    # x1, y1, x2, y2 = lines[0]
    # print(x1, y1, x2, y2)
    #