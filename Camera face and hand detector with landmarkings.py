import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                            )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                            )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                            )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                            )

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Check if face and hands are visible
        is_face_visible = results.face_landmarks is not None
        is_left_hand_visible = results.left_hand_landmarks is not None
        is_right_hand_visible = results.right_hand_landmarks is not None

        # Display text with background
        text_padding = 10
        if is_face_visible:
            text = "Face"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            text_width, text_height = text_size[0], text_size[1]
            cv2.rectangle(image, (text_padding, text_padding), (text_padding + text_width, text_padding + text_height), (0, 0, 0, 76), cv2.FILLED)
            cv2.putText(image, text, (text_padding, text_padding + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if is_left_hand_visible:
            text = "Left Hand"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            text_width, text_height = text_size[0], text_size[1]
            cv2.rectangle(image, (text_padding, text_padding + text_height + 20), (text_padding + text_width, text_padding + text_height * 2 + 20), (0, 0, 0, 76), cv2.FILLED)
            cv2.putText(image, text, (text_padding, text_padding + text_height * 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if is_right_hand_visible:
            text = "Right Hand"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            text_width, text_height = text_size[0], text_size[1]
            cv2.rectangle(image, (text_padding, text_padding + text_height * 3 + 40), (text_padding + text_width, text_padding + text_height * 4 + 40), (0, 0, 0, 76), cv2.FILLED)
            cv2.putText(image, text, (text_padding, text_padding + text_height * 4 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Show image
        cv2.imshow('OpenCV Feed', image)

        # Check for 'q' or window close event
        if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty('OpenCV Feed', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
