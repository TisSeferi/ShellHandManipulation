import cv2
import mediapipe as mp
import modin.pandas as pd
import time
import DataManagement as dm

class HandDetecter:

    def __init__(self, DataManagement):
        self.cap = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.prevTime = 0
        self.cTime = 0
        self.DataGuy = dm.DataGuy()

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                continue

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    self.DataGuy.from_landmark(handLms.landmark)
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 0:
                            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            self.currTime = time.time()
            fps = 1 / (self.currTime - self.prevTime)
            self.prevTime = self.currTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)

            if (cv2.waitKey(1) & 0xFF ==27):
                break

        self.cap.release()
        cv2.destroyAllWindows()