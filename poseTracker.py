import cv2
import mediapipe as mp
import time


class poseTracker():
    def __init__(self) -> None:
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def track(self, img):
        # convert OpenCV's BGR image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.pose.process(imgRGB)

    def Draw(self, results, img):
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    tracker = poseTracker()

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        results = poseTracker.track(tracker, img)

        img = poseTracker.Draw(tracker, results, img)

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
