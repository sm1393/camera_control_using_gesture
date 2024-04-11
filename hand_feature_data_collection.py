import cv2
import mediapipe as mp
from openpyxl import load_workbook
import numpy as np
import math

filePath = "dataset/hand_features_data.xlsx"
wb = load_workbook(filePath)
ws = wb.worksheets[0]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

collection = 0
while True:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                normalizedOriginX = hand_landmarks.landmark[0].x
                normalizedOriginY = hand_landmarks.landmark[0].y
                rBase = np.sqrt((hand_landmarks.landmark[1].x - normalizedOriginX)**2 + (hand_landmarks.landmark[1].y - normalizedOriginY)**2)
                thetaBase = math.atan2(hand_landmarks.landmark[1].y - normalizedOriginY, hand_landmarks.landmark[1].x - normalizedOriginX)
                counter = 0
                featureArray = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    r = np.sqrt((landmark.x - normalizedOriginX)**2 + (landmark.y - normalizedOriginY)**2)
                    rRatio = r / rBase
                    theta = math.atan2(landmark.y - normalizedOriginY, landmark.x - normalizedOriginX)
                    thetaRatio = theta - thetaBase
                    if counter == 0:
                        counter += 1
                        continue
                    featureArray.append(rRatio)
                    featureArray.append(thetaRatio)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif 97 <= k <= 122:
            featureArray.insert(0, chr(k))
            frame = cv2.putText(frame, chr(k), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 10)
            break
        cv2.imshow('Hand Landmarks', frame)
    cv2.imshow('Hand Landmarks', frame)
    k = cv2.waitKey()
    if k == 27:
        break
    elif k == 115:
        collection += 1
        print(collection, " Accepted")
        ws.append(featureArray)
    elif k == 110:
        print("Rejected")

wb.save(filePath)
cap.release()
cv2.destroyAllWindows()

hands.close()

# s = 115
# n = 110
# esp = 27