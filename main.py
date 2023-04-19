import cv2
import time

import handTracker
import poseTracker


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    pTracker = poseTracker.poseTracker()
    hTracker = handTracker.handTracker()

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        pResults = pTracker.track(img)
        hResults = hTracker.Track(img)

        img = pTracker.Draw(pResults, img)
        img = hTracker.draw(hResults, img)

        # gather FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # display FPS
        cv2.putText(img, str(int(fps)), (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("view", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
