import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):

        # convert OpenCV's BGR image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.hands.process(imgRGB)

    def draw(self, results, img):
        if results.multi_hand_landmarks:  # if tracking
            for handLms in results.multi_hand_landmarks:  # for every tracker point
                # draw it on the image
                self.mpDraw.draw_landmarks(
                    img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = handDetector()

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        results = detector.findHands(img)

        img = detector.draw(results, img)

        # gather FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # display FPS
        cv2.putText(img, str(int(fps)), (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # display the image
        cv2.imshow("view", img)
        cv2.waitKey(1)  # ???


if __name__ == "__main__":
    main()
