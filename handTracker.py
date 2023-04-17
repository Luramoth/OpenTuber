import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def Track(img):
    # convert OpenCV's BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # tell Mediapipe to process the image
    return hands.process(imgRGB)


def Draw(img, results):
    if results.multi_hand_landmarks:  # if tracking
        for handLms in results.multi_hand_landmarks:  # for every tracker point
            # draw it on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


def main():
    # grab the webcam
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    while True:
        # grab the current frame from the webcam
        success, img = cap.read()

        # get the results of tracking
        results = Track(img)

        Draw(img, results)

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
