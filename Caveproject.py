import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

COLORS = {
    'Green':  (0, 255, 0),
    'Blue':   (255, 0, 0),
    'Red':    (0, 0, 255),
    'Yellow': (0, 255, 255),
    'White':  (255, 255, 255)
}
color_names = list(COLORS.keys())
current_color_idx = 0
draw_color = COLORS[color_names[current_color_idx]]
brush_size = 10

canvas = None
prev_x, prev_y = None, None

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Started! Index finger UP = Draw | Two fingers UP = Stop | C=Clear N=Color Q=Quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv.flip(img, 1)
    h, w, _ = img.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]

        # All landmark positions
        index_tip  = hand_landmarks[8]
        index_pip  = hand_landmarks[6]
        middle_tip = hand_landmarks[12]
        middle_pip = hand_landmarks[10]
        ring_tip   = hand_landmarks[16]
        ring_pip   = hand_landmarks[14]
        pinky_tip  = hand_landmarks[20]
        pinky_pip  = hand_landmarks[18]

        # Index fingertip pixel position
        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        # Finger up detection — tip.y smaller than pip.y means finger is UP
        index_up  = (index_pip.y  - index_tip.y)  > 0.04
        middle_up = (middle_pip.y - middle_tip.y) > 0.04
        ring_up   = (ring_pip.y   - ring_tip.y)   > 0.04
        pinky_up  = (pinky_pip.y  - pinky_tip.y)  > 0.04

        # Draw hand skeleton
        for connection in HAND_CONNECTIONS:
            s, e = connection
            sx = int(hand_landmarks[s].x * w)
            sy = int(hand_landmarks[s].y * h)
            ex = int(hand_landmarks[e].x * w)
            ey = int(hand_landmarks[e].y * h)
            cv.line(img, (sx, sy), (ex, ey), (150, 150, 150), 2)

        # Draw all landmark dots
        for idx, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if idx in [4, 8, 12, 16, 20]:
                cv.circle(img, (cx, cy), 8, (255, 0, 255), cv.FILLED)
            else:
                cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)

        # ✌️ Two or more fingers up = STOP drawing
        if index_up and middle_up:
            prev_x, prev_y = None, None
            cv.circle(img, (ix, iy), 20, (0, 0, 255), 3)
            cv.putText(img, "HOVER", (ix + 20, iy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ☝️ Only index finger up = DRAW
        elif index_up and not middle_up and not ring_up and not pinky_up:
            cv.circle(img, (ix, iy), brush_size, draw_color, cv.FILLED)
            cv.putText(img, "DRAW", (ix + 20, iy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)
            if prev_x is not None and prev_y is not None:
                cv.line(canvas, (prev_x, prev_y), (ix, iy), draw_color, brush_size)
            prev_x, prev_y = ix, iy

        else:
            prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None
        cv.putText(img, "Show your hand!", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Merge drawing canvas onto camera frame
    canvas_gray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(canvas_gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    img_bg   = cv.bitwise_and(img, img, mask=mask_inv)
    canvas_fg = cv.bitwise_and(canvas, canvas, mask=mask)
    combined  = cv.add(img_bg, canvas_fg)

    # Top toolbar
    cv.rectangle(combined, (0, 0), (w, 55), (30, 30, 30), cv.FILLED)
    swatch_w = w // len(color_names)
    for i, name in enumerate(color_names):
        color = COLORS[name]
        x1 = i * swatch_w
        x2 = x1 + swatch_w
        cv.rectangle(combined, (x1 + 5, 6), (x2 - 5, 48), color, cv.FILLED)
        if i == current_color_idx:
            cv.rectangle(combined, (x1 + 3, 4), (x2 - 3, 50), (255, 255, 255), 3)
        cv.putText(combined, name, (x1 + 8, 44),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Bottom info bar
    cv.rectangle(combined, (0, h - 35), (w, h), (30, 30, 30), cv.FILLED)
    cv.putText(combined,
               f"Color: {color_names[current_color_idx]} | Brush: {brush_size} | C=Clear  N=Next Color  +=Bigger  -=Smaller  Q=Quit",
               (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv.imshow("Finger Drawing App", combined)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        print("Canvas cleared!")
    elif key == ord('n'):
        current_color_idx = (current_color_idx + 1) % len(color_names)
        draw_color = COLORS[color_names[current_color_idx]]
        print(f"Color: {color_names[current_color_idx]}")
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 2, 50)
        print(f"Brush size: {brush_size}")
    elif key == ord('-'):
        brush_size = max(brush_size - 2, 2)
        print(f"Brush size: {brush_size}")

cap.release()
cv.destroyAllWindows()
print("App closed.")