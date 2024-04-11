import cv2
import mediapipe as mp
import numpy as np
import math
import joblib

label = ['l', 'o', 'v', 'z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
pipeline = joblib.load('my_pipeline.joblib')

cap = cv2.VideoCapture(0)
featureArray = list()

def zoom_at_point(image, zoom_factor, zoom_point):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(zoom_point, 0, zoom_factor)
    zoomed_image = cv2.warpAffine(image, matrix, (width, height))
    return zoomed_image

zoomFactor = 0
zoomCenterX = 0
zoomCenterY = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            normalizedOriginX = hand_landmarks.landmark[0].x
            normalizedOriginY = hand_landmarks.landmark[0].y
            rBase = np.sqrt((hand_landmarks.landmark[1].x - normalizedOriginX)**2 + (hand_landmarks.landmark[1].y - normalizedOriginY)**2)
            thetaBase = math.atan2(hand_landmarks.landmark[1].y - normalizedOriginY, hand_landmarks.landmark[1].x - normalizedOriginX)
            featureArray = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                r = np.sqrt((landmark.x - normalizedOriginX)**2 + (landmark.y - normalizedOriginY)**2)
                rRatio = r / rBase
                theta = math.atan2(landmark.y - normalizedOriginY, landmark.x - normalizedOriginX)
                thetaRatio = theta - thetaBase
                featureArray.append(rRatio)
                featureArray.append(thetaRatio)
            zoomCenterX = int(hand_landmarks.landmark[8].x*frame.shape[1])
            zoomCenterY = int(hand_landmarks.landmark[8].y*frame.shape[0])
    if len(featureArray) == 42:
        featureArray = np.array(featureArray[4:])
        output = pipeline.predict_proba(featureArray.reshape(1, -1))
        prediction = np.argmax(output, axis=1)[0]
        if np.round(output[0][prediction], 5) > 0.999:
            if label[prediction] == 'l':
                zoomFactor += 0.1
            if label[prediction] == 'o':
                zoomFactor -= 0.1
            if zoomFactor < 0:
                zoomFactor = 0 
            if zoomFactor > 3:
                zoomFactor = 3
    zoomed_image = zoom_at_point(frame, 1+zoomFactor, (zoomCenterX, zoomCenterY))
    cv2.imshow('Hand Landmarks', frame)
    cv2.imshow('zoomed_image', zoomed_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

hands.close()