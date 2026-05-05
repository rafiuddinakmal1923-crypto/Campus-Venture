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

# Define connections manually (replaces solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17)                 # Palm
]

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, c = img.shape
            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)

                if idx == 8:  # Index finger tip
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv.line(img, start_point, end_point, (0, 255, 0), 2)

    cv.imshow("Hand Tracker", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()