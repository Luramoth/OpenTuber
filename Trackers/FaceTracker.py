import cv2
import mediapipe as mp
import time


class FaceTracker:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFace = mp.solutions.face_mesh
        self.face = self.mpFace.FaceMesh()

    def track(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.face.process(img_rgb)

    def draw(self, results, img):

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms)

        return img


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    p_time = 0
    c_time = 0

    tracker = FaceTracker()

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        results = FaceTracker.track(tracker, img)

        img = FaceTracker.draw(tracker, results, img)

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
