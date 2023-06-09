import cv2
import mediapipe as mp
import time


class PoseTracker:
    def __init__(self) -> None:
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(model_complexity=0)
        self.mpDraw = mp.solutions.drawing_utils

    def track(self, img):
        # convert OpenCV's BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.pose.process(img_rgb)

    def draw(self, results, img):
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


# noinspection PyTypeChecker
def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    p_time = 0

    tracker = PoseTracker()

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        results = PoseTracker.track(tracker, img)

        img = PoseTracker.draw(tracker, results, img)

        # gather FPS
        c_time = time.time()
        fps = 1 / (cTime - p_time)
        p_time = c_time

        # display FPS
        cv2.putText(img, str(int(fps)), (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("view", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
