"""
Fall Guys – Hand-Gesture Controller (Head Tilt Edition)
=================================================
HAND GESTURES → Movement & Actions:
    Forward    (W)     → Right hand OPEN (all fingers up)
    Backward   (S)     → Right hand INDEX only
    Right      (D)     → Left hand INDEX only
    Left       (A)     → Left hand PEACE sign (Index & Middle)
    Jump       (Space) → BOTH hands OPEN
    Grab       (Shift) → Tight FIST on either hand

POSE (arms) → Dive (stretch both arms wide) → Ctrl
POSE (head) → Camera rotation (tilt head)   → Mouse drag

NOTE ON MIRRORING:
    cv.flip(frame, 1) mirrors the preview so it feels natural to the user.
    After flipping, MediaPipe labels are inverted:
        MediaPipe 'Left'  → user's RIGHT hand
        MediaPipe 'Right' → user's LEFT hand
        MediaPipe LEFT_EAR  → user's RIGHT ear
        MediaPipe RIGHT_EAR → user's LEFT ear
    All corrections are applied consistently below.

Requirements:
    pip install opencv-python mediapipe pynput

Press Q in the webcam window to quit.
"""

import cv2 as cv
import mediapipe as mp
import time
import threading
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

# ── Input controllers ─────────────────────────────────────────────────────────
keyboard = KeyboardController()
mouse    = MouseController()

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    smooth_landmarks=True,
)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ── Hand skeleton connections ─────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ── Key state tracker ─────────────────────────────────────────────────────────
held_keys = set()

def press_key(k):
    if k not in held_keys:
        keyboard.press(k)
        held_keys.add(k)

def release_key(k):
    if k in held_keys:
        keyboard.release(k)
        held_keys.discard(k)

def release_all_keys():
    for k in list(held_keys):
        keyboard.release(k)
    held_keys.clear()

# ── Camera drag thread ────────────────────────────────────────────────────────
camera_thread = None
camera_active = False
camera_dir    = None
DRAG_AMOUNT   = 8      # pixels per tick — increase for faster rotation
DRAG_INTERVAL = 0.016  # ~60 ticks/s

def _drag_loop():
    global camera_active, camera_dir
    mouse.press(Button.right)
    while camera_active:
        dx = -DRAG_AMOUNT if camera_dir == 'left' else DRAG_AMOUNT
        mouse.move(dx, 0)
        time.sleep(DRAG_INTERVAL)
    mouse.release(Button.right)

def start_camera(direction):
    global camera_thread, camera_active, camera_dir
    if camera_active and camera_dir == direction:
        return
    stop_camera()
    camera_active = True
    camera_dir    = direction
    camera_thread = threading.Thread(target=_drag_loop, daemon=True)
    camera_thread.start()

def stop_camera():
    global camera_active
    camera_active = False
    if camera_thread:
        camera_thread.join(timeout=0.1)

# ── Exponential moving average ────────────────────────────────────────────────
class EMA:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.val   = None
    def update(self, v):
        self.val = v if self.val is None else self.alpha * v + (1 - self.alpha) * self.val
        return self.val

dive_s = EMA(0.35)
head_s = EMA(0.35)

# ── Thresholds ────────────────────────────────────────────────────────────────
DIVE_WRIST_SPREAD   = 0.55   # wrist X spread (0–1 normalised) → dive
HEAD_TILT_THRESHOLD = 0.02   # ear Y difference to trigger camera

# ── Debug mode ────────────────────────────────────────────────────────────────
DEBUG = True

# ── Hand gesture classifier ───────────────────────────────────────────────────
def finger_extended(lms, tip, pip):
    return lms.landmark[tip].y < lms.landmark[pip].y - 0.02

def thumb_extended(lms):
    t   = lms.landmark[4]
    mcp = lms.landmark[2]
    return abs(t.x - mcp.x) > 0.06 or abs(t.y - mcp.y) > 0.06

def get_hand_state(lms):
    thumb  = thumb_extended(lms)
    index  = finger_extended(lms, 8,  6)
    middle = finger_extended(lms, 12, 10)
    ring   = finger_extended(lms, 16, 14)
    pinky  = finger_extended(lms, 20, 18)

    fingers_up_count = sum([index, middle, ring, pinky])

    if thumb and fingers_up_count == 0:                  return 'THUMBS_UP'
    if index and fingers_up_count == 1:                  return 'INDEX'
    if index and middle and fingers_up_count == 2:       return 'PEACE'
    if fingers_up_count >= 3:                            return 'OPEN'
    if not thumb and fingers_up_count == 0:              return 'FIST'
    return 'OTHER'

# ── Draw clean hand skeleton ──────────────────────────────────────────────────
def draw_hand(img, lms, color):
    h, w, _ = img.shape
    for s_idx, e_idx in HAND_CONNECTIONS:
        s = lms.landmark[s_idx]
        e = lms.landmark[e_idx]
        cv.line(img,
                (int(s.x * w), int(s.y * h)),
                (int(e.x * w), int(e.y * h)),
                color, 2)
    for idx, pt in enumerate(lms.landmark):
        cx, cy = int(pt.x * w), int(pt.y * h)
        r = 6 if idx in (0, 5, 9, 13, 17) else 4
        cv.circle(img, (cx, cy), r, color, cv.FILLED)

# ── Head tilt indicator ───────────────────────────────────────────────────────
def draw_head_indicator(img, head_val, threshold):
    h, w, _ = img.shape
    cx, cy = w - 70, 200
    r = 38
    cv.circle(img, (cx, cy), r, (40, 40, 40), -1)
    cv.circle(img, (cx, cy), r, (80, 80, 80), 2)
    clamped = max(-1.0, min(1.0, head_val / (threshold * 2.5)))
    dot_x   = int(cx - clamped * r * 0.75)  # negated: dot follows user's tilt direction
    dot_col = (220, 180, 50) if abs(head_val) > threshold else (160, 160, 160)
    cv.circle(img, (dot_x, cy), 10, dot_col, -1)
    # positive head_val = user tilts left = camera left
    turn = 'L' if head_val > threshold else ('R' if head_val < -threshold else '-')
    cv.putText(img, f"CAM {turn}", (cx - 28, cy + r + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.44, dot_col, 1)

# ── HUD ───────────────────────────────────────────────────────────────────────
COLORS = {
    'move': (50,  220,  50),
    'jump': (50,  220, 255),
    'dive': (255, 150,  50),
    'grab': (0,   100, 255),
    'cam':  (220, 180,  50),
    'idle': (160, 160, 160),
}

def draw_hud(img, body_text, hand_action, cam_action, fps,
             left_state, right_state, head_val, dive_val):
    h, w, _ = img.shape

    # top bar
    overlay = img.copy()
    cv.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    bc = (COLORS['jump'] if 'JUMP' in body_text else
          COLORS['dive'] if 'DIVE' in body_text else
          COLORS['idle'] if body_text == 'NEUTRAL' else
          COLORS['move'])
    cv.putText(img, f"MOVE:  {body_text}", (14, 36),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, bc, 2)

    cc = COLORS['cam'] if cam_action != 'FORWARD' else COLORS['idle']
    cv.putText(img, f"HEAD:  {cam_action}", (14, 66),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, cc, 2)

    hc = COLORS['grab'] if hand_action == 'GRAB' else COLORS['idle']
    cv.putText(img, f"HAND:  {hand_action}", (14, 96),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, hc, 2)

    # bottom bar
    bot = img.copy()
    cv.rectangle(bot, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv.addWeighted(bot, 0.45, img, 0.55, 0, img)
    cv.putText(img, f"FPS {fps:.0f}   |   Q to quit   |   Alt-tab into Fall Guys",
               (12, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

    # debug panel (top-right)
    if DEBUG:
        x, y = w - 230, 130
        lines = [
            f"L hand: {left_state}",
            f"R hand: {right_state}",
            f"head:   {head_val:+.4f} (thr {HEAD_TILT_THRESHOLD})",
            f"dive:   {dive_val:.4f}  (thr 0.6)",
        ]
        cv.rectangle(img, (x - 6, y - 14), (w - 4, y + len(lines) * 18 + 4),
                     (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv.putText(img, line, (x, y + i * 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

# ── Main loop ─────────────────────────────────────────────────────────────────

# ── Camera selection ──────────────────────────────────────────────────────────
# 0 = built-in laptop camera
# 1 = first external webcam  <- try this first
# 2 = second external webcam (if 1 doesn't work)
CAMERA_INDEX = 1

cap = cv.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Camera {CAMERA_INDEX} not found. Scanning available cameras...")
    for i in range(5):
        test = cv.VideoCapture(i)
        if test.isOpened():
            print(f"  Camera index {i} is available")
            test.release()
    print("Update CAMERA_INDEX at the top of the script to match yours.")
    print("Falling back to camera 0.\n")
    cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)

# ── Resizable window ──────────────────────────────────────────────────────────
cv.namedWindow('Fall Guys Controller', cv.WINDOW_NORMAL)
cv.resizeWindow('Fall Guys Controller', 640, 480)

print("=== Fall Guys Gesture Controller ===")
print("  Forward (W)   → Right hand OPEN")
print("  Backward (S)  → Right hand INDEX finger")
print("  Right (D)     → Left hand INDEX finger")
print("  Left (A)      → Left hand PEACE sign")
print("  Jump (Space)  → BOTH hands OPEN")
print("  Grab (Shift)  → Tight FIST (either hand)")
print("  Dive (Ctrl)   → Stretch arms wide")
print("  Camera        → Tilt head left / right")
print("\nPress Q to quit.\n")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the preview — makes gestures feel natural
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    pose_results  = pose.process(rgb)
    hands_results = hands.process(rgb)

    now       = time.time()
    fps       = 1 / (now - prev_time + 1e-9)
    prev_time = now

    cam_action       = 'FORWARD'
    hand_action      = 'IDLE'
    body_text        = 'NEUTRAL'
    left_hand_state  = 'NONE'
    right_hand_state = 'NONE'
    head_val         = head_s.val or 0.0
    dive_val         = dive_s.val or 0.0

    # ── 1. HANDS ─────────────────────────────────────────────────────────────
    if hands_results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(hands_results.multi_hand_landmarks):
            label = hands_results.multi_handedness[idx].classification[0].label

            # Frame is flipped BEFORE sending to MediaPipe, so MediaPipe sees
            # the already-mirrored image. This means its labels are already
            # correct from the user's perspective — no swap needed.
            actual_hand = label  # 'Left' = user's left, 'Right' = user's right
            state = get_hand_state(hand_lms)

            if actual_hand == 'Left':
                left_hand_state = state
                color = (255, 150, 50)    # orange = left hand
            else:
                right_hand_state = state
                color = (50, 255, 150)    # green  = right hand

            if state == 'FIST':
                color = (0, 100, 255)     # blue override for fist

            draw_hand(frame, hand_lms, color)
            cv.putText(frame, f"{actual_hand}: {state}",
                       (int(hand_lms.landmark[0].x * w),
                        int(hand_lms.landmark[0].y * h) + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── 2. POSE: dive + head tilt ─────────────────────────────────────────────
    if pose_results.pose_landmarks:
        lms = pose_results.pose_landmarks.landmark

        # Dive — wrists spread wide
        lw_v = lms[mp_pose.PoseLandmark.LEFT_WRIST].visibility
        rw_v = lms[mp_pose.PoseLandmark.RIGHT_WRIST].visibility
        lw_x = lms[mp_pose.PoseLandmark.LEFT_WRIST].x
        rw_x = lms[mp_pose.PoseLandmark.RIGHT_WRIST].x
        wrist_spread = abs(lw_x - rw_x) if (lw_v > 0.3 and rw_v > 0.3) else 0.0
        dive_val = dive_s.update(1.0 if wrist_spread > DIVE_WRIST_SPREAD else 0.0)

        # Head tilt — ear Y difference
        # Frame is flipped before MediaPipe processes it, so landmarks
        # are already from the user's perspective — no swap needed.
        # User tilts LEFT  → left ear drops  → LEFT_EAR Y increases  → tilt_raw positive → cam LEFT
        # User tilts RIGHT → right ear drops → RIGHT_EAR Y increases → tilt_raw negative → cam RIGHT
        user_left_ear_y  = lms[mp_pose.PoseLandmark.RIGHT_EAR].y
        user_right_ear_y = lms[mp_pose.PoseLandmark.LEFT_EAR].y
        tilt_raw = user_left_ear_y - user_right_ear_y
        head_val = head_s.update(tilt_raw)

        if head_val > HEAD_TILT_THRESHOLD:
            cam_action = 'LEFT'
            start_camera('left')
        elif head_val < -HEAD_TILT_THRESHOLD:
            cam_action = 'RIGHT'
            start_camera('right')
        else:
            cam_action = 'FORWARD'
            stop_camera()

        draw_head_indicator(frame, head_val, HEAD_TILT_THRESHOLD)

    else:
        stop_camera()

    # ── 3. ACTION RESOLUTION ─────────────────────────────────────────────────

    # Dive — top priority
    if dive_val > 0.6:
        body_text = 'DIVE'
        release_key('w'); release_key('s')
        release_key('a'); release_key('d')
        release_key(Key.space)
        press_key(Key.ctrl_l)

    # Jump — both hands open
    elif left_hand_state == 'OPEN' and right_hand_state == 'OPEN':
        body_text = 'JUMP'
        release_key('w'); release_key('s')
        release_key('a'); release_key('d')
        release_key(Key.ctrl_l)
        press_key(Key.space)

    # Movement — gestures can combine (e.g. FWD + LEFT simultaneously)
    else:
        release_key(Key.space)
        release_key(Key.ctrl_l)

        active_moves = []

        # Forward / backward on RIGHT hand
        if right_hand_state == 'OPEN':
            press_key('w'); active_moves.append('FWD')
        else:
            release_key('w')

        if right_hand_state == 'INDEX':
            press_key('s'); active_moves.append('BACK')
        else:
            release_key('s')

        # Left / right on LEFT hand
        if left_hand_state == 'PEACE':
            press_key('a'); active_moves.append('LEFT')
        else:
            release_key('a')

        if left_hand_state == 'INDEX':
            press_key('d'); active_moves.append('RIGHT')
        else:
            release_key('d')

        body_text = ' + '.join(active_moves) if active_moves else 'NEUTRAL'

    # Grab — fist on either hand, independent of movement
    if left_hand_state == 'FIST' or right_hand_state == 'FIST':
        hand_action = 'GRAB'
        press_key(Key.shift_l)
    else:
        hand_action = 'IDLE'
        release_key(Key.shift_l)

    draw_hud(frame, body_text, hand_action, cam_action, fps,
             left_hand_state, right_hand_state, head_val, dive_val)
    cv.imshow('Fall Guys Controller', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

release_all_keys()
stop_camera()
cap.release()
cv.destroyAllWindows()
print("Controller stopped. All keys released.")