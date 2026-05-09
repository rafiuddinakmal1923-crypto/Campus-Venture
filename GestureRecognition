import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Define connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17)                 # Palm
]

cap = cv.VideoCapture(0)

def is_finger_extended(landmarks, tip_idx, pip_idx):
    """Check if a finger is extended"""
    return landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    results = detector.detect(mp_image)

    gesture_text = "No Hand Detected"

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, c = img.shape
            
            # Draw landmarks and connections
            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
                if idx == 8:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv.line(img, start_point, end_point, (0, 255, 0), 2)

            # ====================== GESTURE DETECTION ======================
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            thumb_pt = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pt = (int(index_tip.x * w), int(index_tip.y * h))
            
            dist_thumb_index = ((thumb_pt[0] - index_pt[0])**2 + (thumb_pt[1] - index_pt[1])**2) ** 0.5

            # Finger states
            thumb_extended = hand_landmarks[4].x < hand_landmarks[3].x - 0.05 if hand_landmarks[4].x < hand_landmarks[0].x else hand_landmarks[4].x > hand_landmarks[3].x + 0.05
            index_ext = is_finger_extended(hand_landmarks, 8, 6)
            middle_ext = is_finger_extended(hand_landmarks, 12, 10)
            ring_ext = is_finger_extended(hand_landmarks, 16, 14)
            pinky_ext = is_finger_extended(hand_landmarks, 20, 18)

            # Gesture Classification
            if dist_thumb_index < 35:
                gesture_text = "PINCH / OK"
                cv.putText(img, "PINCH / OK", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                cv.circle(img, thumb_pt, 15, (0, 0, 255), cv.FILLED)
                cv.circle(img, index_pt, 15, (0, 0, 255), cv.FILLED)
                print("✅ PINCH / OK detected")

            elif thumb_extended and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
                gesture_text = "THUMBS UP"
                cv.putText(img, "THUMBS UP 👍", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                print("✅ THUMBS UP detected")

            elif index_ext and middle_ext and not ring_ext and not pinky_ext:
                gesture_text = "PEACE / TWO"
                cv.putText(img, "PEACE ✌️", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1.3, (255, 165, 0), 3)
                print("✅ PEACE / TWO detected")

            elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
                gesture_text = "POINTING"
                cv.putText(img, "POINTING 👆", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 3)
                print("✅ POINTING detected")

            elif not index_ext and not middle_ext and not ring_ext and not pinky_ext:
                gesture_text = "FIST / ROCK"
                cv.putText(img, "FIST ✊", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)
                print("✅ FIST / ROCK detected")

            elif index_ext and middle_ext and ring_ext and pinky_ext:
                gesture_text = "OPEN HAND / FIVE"
                cv.putText(img, "OPEN HAND ✋", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
                print("✅ OPEN HAND detected")

            else:
                gesture_text = "Unknown Gesture"
                cv.putText(img, "Unknown Gesture", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show current gesture at bottom
            cv.putText(img, f"Gesture: {gesture_text}", (50, 450), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    else:
        cv.putText(img, "No Hand Detected", (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Hand Gesture Recognizer", img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
