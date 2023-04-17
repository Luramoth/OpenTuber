import cv2
import mediapipe as mp
import time

# grab image out of webcam
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()  # grab frame from webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change it to RGB for MP
    results = hands.process(imgRGB)  # have MP try to find the hands

    # display trackers
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

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
