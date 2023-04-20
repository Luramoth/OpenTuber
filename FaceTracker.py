import cv2
import mediapipe as mp
import time


class FaceTracker:
    pass


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    p_time = 0
    c_time = 0

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        # gather FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # display FPS
        cv2.putText(img, str(int(fps)), (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # display the image
        cv2.imshow("view", img)
        cv2.waitKey(1)  # ???


if __name__ == "__main__":
    main()
