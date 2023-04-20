import cv2
import time

from Trackers import FaceTracker, HandTracker, PoseTracker


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    p_time = 0

    p_tracker = PoseTracker.PoseTracker()
    h_tracker = HandTracker.HandTracker()
    f_tracker = FaceTracker.FaceTracker()

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        p_results = p_tracker.track(img)
        h_results = h_tracker.track(img)
        f_results = f_tracker.track(img)

        img = p_tracker.draw(p_results, img)
        img = h_tracker.draw(h_results, img)
        img = f_tracker.draw(f_results, img)

        # gather FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # display FPS
        cv2.putText(img, str(int(fps)), (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("view", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
