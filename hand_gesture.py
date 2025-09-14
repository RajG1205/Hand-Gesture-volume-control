import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math
import time

# Hand Gesture Volume Control

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize Audio Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Open webcam
cap = cv2.VideoCapture(0)
print("Hand Gesture Volume Control started. Press 'q' to quit.")

# Finger detection helper
def get_fingers(hand_landmarks):
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb check (x-axis)
    fingers.append(landmarks[tips[0]][0] < landmarks[tips[0] - 1][0])

    # Other fingers check (y-axis)
    for tip in tips[1:]:
        fingers.append(landmarks[tip][1] < landmarks[tip - 2][1])

    return fingers

# Cooldown timer for volume changes
last_change_time = 0
cooldown = 0.3  # seconds between volume updates
step = 0.02     # 2% volume change per step

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture = ""
    current_vol = volume.GetMasterVolumeLevelScalar()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_fingers(hand_landmarks)
            total = fingers.count(True)

            now = time.time()

            # Gesture recognition
            if total == 0:
                gesture = "Mute"
                volume.SetMute(1, None)

            elif total == 5 and (now - last_change_time) > cooldown:
                gesture = "Open Palm → Volume Up"
                volume.SetMute(0, None)
                volume.SetMasterVolumeLevelScalar(min(1.0, current_vol + step), None)
                last_change_time = now

            elif total == 1 and fingers[1] and (now - last_change_time) > cooldown:
                gesture = "Pointing → Volume Down"
                volume.SetMute(0, None)
                volume.SetMasterVolumeLevelScalar(max(0.0, current_vol - step), None)
                last_change_time = now

            else:
                # Smooth Thumb–Index control
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                h, w, _ = frame.shape
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                distance = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(distance, [30, 200], [0.0, 1.0])
                volume.SetMasterVolumeLevelScalar(vol, None)
                gesture = f"Thumb-Index Control ({int(vol * 100)}%)"

    # Draw volume bar with better style
    vol_level = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.rectangle(frame, (50, 150), (85, 400), (200, 200, 200), 3, cv2.LINE_AA)
    cv2.rectangle(frame, (50, int(400 - (vol_level * 2.5))), (85, 400), (0, 200, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, f"{vol_level}%", (40, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 3, cv2.LINE_AA)

    # Display gesture text
    if gesture:
        color = (0, 0, 255) if gesture == "Mute" else (0, 200, 0)
        cv2.putText(frame, gesture, (120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
