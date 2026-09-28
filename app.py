import streamlit as st
import time
import math
import cv2
import numpy as np
from datetime import datetime
from collections import deque

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False

st.set_page_config(
    page_title="Gesture Smart Lock Simulator",
    page_icon="🚪",
    layout="wide"
)

# ----------------------------
# Constants
# ----------------------------
FRAME_W, FRAME_H = 640, 480

DOOR_LEFT = 185
DOOR_RIGHT = 455
DOOR_TOP = 85
DOOR_BOTTOM = 395

PASSCODE = ("fist", "open_palm", "point")
WAVE_WINDOW = 1.0
SEQ_WINDOW = 4.0
OPEN_DURATION = 0.8
LOCKOUT_ATTEMPTS = 3
LOCKOUT_WINDOW = 60.0
LOCKOUT_DURATION = 30.0
GUEST_HOLD = 1.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ----------------------------
# Canonical synthetic hands
# ----------------------------
CANONICAL_HANDS = {}


def _build_canonical_hands():
    CANONICAL_HANDS["open_palm"] = np.array([
        [0.50, 0.82, 0.0],  # 0 wrist
        [0.44, 0.76, 0.0],  # 1 thumb cmc
        [0.40, 0.71, 0.0],  # 2
        [0.37, 0.66, 0.0],  # 3
        [0.35, 0.61, 0.0],  # 4 thumb tip
        [0.46, 0.63, 0.0],  # 5 index mcp
        [0.45, 0.54, 0.0],  # 6 index pip
        [0.44, 0.46, 0.0],  # 7 index dip
        [0.43, 0.38, 0.0],  # 8 index tip
        [0.50, 0.61, 0.0],  # 9 middle mcp
        [0.50, 0.51, 0.0],  # 10
        [0.50, 0.42, 0.0],  # 11
        [0.50, 0.34, 0.0],  # 12
        [0.54, 0.63, 0.0],  # 13 ring mcp
        [0.55, 0.54, 0.0],  # 14
        [0.56, 0.46, 0.0],  # 15
        [0.57, 0.39, 0.0],  # 16
        [0.58, 0.66, 0.0],  # 17 pinky mcp
        [0.60, 0.58, 0.0],  # 18
        [0.61, 0.51, 0.0],  # 19
        [0.62, 0.45, 0.0],  # 20
    ], dtype=float)

    CANONICAL_HANDS["fist"] = np.array([
        [0.50, 0.82, 0.0],
        [0.45, 0.76, 0.0],
        [0.44, 0.72, 0.0],
        [0.45, 0.69, 0.0],
        [0.47, 0.68, 0.0],
        [0.46, 0.67, 0.0],
        [0.47, 0.65, 0.0],
        [0.47, 0.64, 0.0],
        [0.47, 0.64, 0.0],
        [0.50, 0.66, 0.0],
        [0.50, 0.64, 0.0],
        [0.50, 0.63, 0.0],
        [0.50, 0.63, 0.0],
        [0.54, 0.67, 0.0],
        [0.54, 0.65, 0.0],
        [0.54, 0.64, 0.0],
        [0.54, 0.64, 0.0],
        [0.57, 0.69, 0.0],
        [0.57, 0.67, 0.0],
        [0.57, 0.66, 0.0],
        [0.57, 0.66, 0.0],
    ], dtype=float)

    CANONICAL_HANDS["point"] = np.array([
        [0.50, 0.82, 0.0],
        [0.44, 0.76, 0.0],
        [0.42, 0.72, 0.0],
        [0.41, 0.69, 0.0],
        [0.42, 0.67, 0.0],
        [0.46, 0.64, 0.0],
        [0.45, 0.54, 0.0],
        [0.45, 0.46, 0.0],
        [0.45, 0.38, 0.0],
        [0.50, 0.66, 0.0],
        [0.50, 0.64, 0.0],
        [0.50, 0.63, 0.0],
        [0.50, 0.63, 0.0],
        [0.54, 0.67, 0.0],
        [0.54, 0.65, 0.0],
        [0.54, 0.64, 0.0],
        [0.54, 0.64, 0.0],
        [0.57, 0.69, 0.0],
        [0.57, 0.67, 0.0],
        [0.57, 0.66, 0.0],
        [0.57, 0.66, 0.0],
    ], dtype=float)


_build_canonical_hands()

# ----------------------------
# Demo timeline
# ----------------------------
DEMO_POSES = [
    (2.0, 3.3, "open_palm", 1, {"wrist_x": 0.50}),
    (3.5, 5.0, "wave", 1, {}),
    (6.0, 10.0, "open_palm", 1, {"wrist_x": 0.55}),

    (20.5, 21.5, "fist", 1, {}),
    (21.8, 22.8, "open_palm", 1, {}),
    (23.1, 24.1, "point", 1, {}),

    (26.0, 26.8, "fist", 1, {}),
    (27.0, 27.8, "fist", 1, {}),
    (28.0, 28.8, "point", 1, {}),

    (30.0, 30.8, "fist", 1, {}),
    (31.0, 31.8, "open_palm", 1, {}),
    (32.0, 32.8, "fist", 1, {}),

    (34.0, 34.8, "point", 1, {}),
    (35.0, 35.8, "open_palm", 1, {}),
    (36.0, 36.8, "fist", 1, {}),

    (68.0, 69.2, "crossed", 2, {}),
    (71.0, 72.5, "wave", 1, {}),
    (73.0, 76.0, "open_palm", 1, {"wrist_x": 0.55}),
]

DEMO_ACTIONS = [
    (19.5, "secure_on"),
    (78.0, "stop"),
]

DEMO_END = 78.0

# ----------------------------
# Session state helpers
# ----------------------------
def reset_core_state(demo=True):
    ss = st.session_state
    ss.sim_time = 0.0
    ss.last_loop_real = time.time()
    ss.running = True
    ss.paused = False

    ss.secure_mode = False
    ss.guest_mode_until = 0.0
    ss.lockout_until = 0.0
    ss.incorrect_times = []

    ss.gesture_seq = deque()
    ss.wave_history = deque()
    ss.crossed_since = None

    ss.door_angle = 0.0
    ss.door_target = 0.0
    ss.lock_state = "locked"
    ss.flash_until = 0.0
    ss.open_time = None

    ss.last_presence_time = 0.0
    ss.presence_now = False

    ss.access_log = []
    ss.status_msg = (
        "Guided demo started. Door is closed and locked."
        if demo else
        "System reset. Door is closed and locked."
    )

    ss.demo_active = demo
    ss.demo_start_sim = 0.0
    ss.demo_actions_done = set()
    ss.manual_queue = deque()

    ss.pending_raw = None
    ss.pending_since = 0.0
    ss.last_registered = None
    ss.last_registered_time = -10.0
    ss.last_wave_time = -10.0

    ss.frame_counter = 0
    ss.cap_error = None


def init_state():
    ss = st.session_state
    if "initialized" not in ss:
        ss.initialized = True

        ss.source = "Simulation"
        ss.time_scale = 2.0
        ss.auto_close_timeout = 5.0
        ss.hard_lock_timeout = 8.0
        ss.guest_duration = 120.0
        ss.wave_min_crossings = 3

        ss.mp_hands = None
        ss.cap = None

        reset_core_state(demo=True)


# ----------------------------
# Callbacks
# ----------------------------
def start_guided_demo():
    reset_core_state(demo=True)
    if "ui_source" in st.session_state:
        st.session_state.ui_source = "Simulation"
    if "ui_speed" in st.session_state:
        st.session_state.ui_speed = 2.0


def reset_system():
    reset_core_state(demo=False)


def toggle_secure_mode():
    ss = st.session_state
    ss.secure_mode = not ss.secure_mode
    ss.demo_active = False
    ss.gesture_seq.clear()
    ss.status_msg = (
        "Secure mode ON: wave-only unlock is rejected. Use passcode or guest mode."
        if ss.secure_mode else
        "Secure mode OFF: wave unlock allowed."
    )


def enable_guest(now, from_button=False):
    ss = st.session_state
    if not from_button and now < ss.lockout_until:
        ss.status_msg = "Lockout active: guest-mode gesture ignored."
        return False
    ss.guest_mode_until = now + ss.guest_duration
    ss.status_msg = f"Guest mode enabled for {int(ss.guest_duration)}s. Wave-only unlock allowed."
    return True


def enable_guest_now():
    ss = st.session_state
    enable_guest(ss.sim_time, from_button=True)
    ss.demo_active = False


def toggle_pause():
    ss = st.session_state
    ss.paused = not ss.paused
    ss.last_loop_real = time.time()


def toggle_run():
    ss = st.session_state
    ss.running = not ss.running
    ss.last_loop_real = time.time()


def queue_manual(pose, duration):
    ss = st.session_state
    now = ss.sim_time
    ss.manual_queue.append((pose, now + duration, now))
    ss.demo_active = False
    ss.status_msg = f"Manual gesture queued: {pose.replace('_', ' ')}"


# ----------------------------
# Geometry / gesture classification
# ----------------------------
def place_hand(base, wrist_x, wrist_y, scale=1.0, mirror=False):
    arr = base.copy()
    if mirror:
        arr[:, 0] = 1.0 - arr[:, 0]
    w = arr[0, :2].copy()
    arr[:, :2] = (arr[:, :2] - w) * scale + np.array([wrist_x, wrist_y])
    arr[:, 2] = 0.0
    return arr


def make_single_hand(pose, local_t, wrist_x=0.5, wrist_y=0.78, scale=1.0, mirror=False):
    if pose == "wave":
        base_name = "open_palm"
        wrist_x = 0.5 + 0.085 * math.sin(2.0 * math.pi * 2.3 * local_t)
    else:
        base_name = pose if pose in CANONICAL_HANDS else "open_palm"

    base = CANONICAL_HANDS[base_name]
    return place_hand(base, wrist_x, wrist_y, scale, mirror)


def classify_shape(lm):
    def d(i, j):
        return float(np.hypot(lm[i, 0] - lm[j, 0], lm[i, 1] - lm[j, 1]))

    extended = 0
    index_extended = d(8, 0) > d(7, 0) * 1.12
    middle_extended = d(12, 0) > d(11, 0) * 1.10
    ring_extended = d(16, 0) > d(15, 0) * 1.10
    pinky_extended = d(20, 0) > d(19, 0) * 1.10
    thumb_extended = d(4, 0) > d(3, 0) * 1.12

    extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

    if extended >= 3 and thumb_extended:
        return "open_palm"

    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "point"

    if extended <= 1 and not thumb_extended:
        return "fist"

    return "none"


def count_zero_crossings(history, min_dx=0.0015):
    if len(history) < 3:
        return 0

    crossings = 0
    prev_sign = None

    for i in range(1, len(history)):
        dx = history[i][1] - history[i - 1][1]
        if abs(dx) < min_dx:
            continue
        sign = 1 if dx > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            crossings += 1
        prev_sign = sign

    return crossings


def update_wave_history(lm, now):
    ss = st.session_state
    ss.wave_history.append((now, float(lm[0, 0])))
    while ss.wave_history and now - ss.wave_history[0][0] > WAVE_WINDOW:
        ss.wave_history.popleft()


def classify_gesture(lm, now):
    ss = st.session_state
    shape = classify_shape(lm)
    if shape == "open_palm":
        crossings = count_zero_crossings(ss.wave_history)
        if crossings >= ss.wave_min_crossings:
            return "wave"
    return shape


def wrists_close(hands):
    if len(hands) < 2:
        return False
    w0 = hands[0][0, :2]
    w1 = hands[1][0, :2]
    return float(np.linalg.norm(w0 - w1)) < 0.10 and abs(float(w0[1] - w1[1])) < 0.08


def update_crossed(hands, now):
    ss = st.session_state
    if wrists_close(hands):
        if ss.crossed_since is None:
            ss.crossed_since = now

        if now - ss.crossed_since >= GUEST_HOLD:
            if now < ss.lockout_until:
                ss.status_msg = "Lockout active: guest-mode gesture ignored."
                ss.crossed_since = None
                return "guest_locked"

            enable_guest(now, from_button=False)
            ss.crossed_since = None
            return "guest_activated"

        return "guest_hold"

    ss.crossed_since = None
    return None


# ----------------------------
# Lock logic
# ----------------------------
def add_access_log(method, event):
    ss = st.session_state
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "method": method,
        "event": event,
    }
    ss.access_log.insert(0, entry)
    ss.access_log = ss.access_log[:5]


def unlock_door(method, now):
    ss = st.session_state

    if now < ss.lockout_until:
        return False

    if (
        ss.lock_state == "unlocked"
        and ss.door_target == 70.0
        and ss.open_time is not None
        and now - ss.open_time < 2.0
    ):
        return False

    ss.lock_state = "unlocked"
    ss.door_target = 70.0
    ss.open_time = now
    ss.last_presence_time = now
    ss.flash_until = 0.0
    ss.incorrect_times = []
    ss.lockout_until = 0.0
    ss.gesture_seq.clear()

    add_access_log(method, "UNLOCK")
    ss.status_msg = f"{method.capitalize()} accepted. Door opening."
    return True


def failed_sequence(now):
    ss = st.session_state
    ss.flash_until = now + 0.8
    ss.incorrect_times = [t for t in ss.incorrect_times if now - t <= LOCKOUT_WINDOW]
    ss.incorrect_times.append(now)

    if len(ss.incorrect_times) >= LOCKOUT_ATTEMPTS:
        ss.lockout_until = now + LOCKOUT_DURATION
        ss.incorrect_times = []
        ss.status_msg = "Too many incorrect passcode sequences. 30s lockout engaged."
    else:
        ss.status_msg = f"Incorrect passcode sequence. Attempt {len(ss.incorrect_times)}/3."

    ss.gesture_seq.clear()


def register_passcode_gesture(g, now):
    ss = st.session_state

    if now < ss.lockout_until:
        return

    if g not in ("fist", "open_palm", "point"):
        return

    seq = ss.gesture_seq
    while seq and now - seq[0][1] > SEQ_WINDOW:
        seq.popleft()

    if seq and seq[-1][0] == g and now - seq[-1][1] < 0.7:
        return

    seq.append((g, now))

    if len(seq) >= 3:
        last3 = list(seq)[-3:]
        if now - last3[0][1] <= SEQ_WINDOW:
            types = tuple(x[0] for x in last3)
            if types == PASSCODE:
                unlock_door("passcode", now)
            else:
                failed_sequence(now)
            seq.clear()


def handle_wave(now):
    ss = st.session_state

    if now < ss.lockout_until:
        ss.status_msg = "Lockout active: wave ignored."
        return

    if now - ss.last_wave_time < 2.0:
        return

    ss.last_wave_time = now
    guest_active = now < ss.guest_mode_until

    if (not ss.secure_mode) or guest_active:
        unlock_door("wave", now)
        if guest_active and ss.secure_mode:
            ss.status_msg = "Guest mode: wave accepted."
    else:
        ss.flash_until = now + 0.6
        ss.status_msg = "Secure mode: wave-only denied. Use passcode or guest mode."


def detect_and_handle_gestures(hands, now, dt_sim):
    ss = st.session_state
    if ss.paused:
        return

    crossed_state = update_crossed(hands, now)
    if crossed_state is not None:
        ss.pending_raw = None
        return

    if not hands:
        ss.pending_raw = None
        return

    update_wave_history(hands[0], now)
    raw = classify_gesture(hands[0], now)

    if raw == "wave":
        handle_wave(now)
        ss.pending_raw = None
        return

    if raw in ("fist", "open_palm", "point"):
        if ss.pending_raw == raw:
            if now - ss.pending_since >= 0.25:
                if ss.last_registered != raw or now - ss.last_registered_time >= 0.8:
                    register_passcode_gesture(raw, now)
                    ss.last_registered = raw
                    ss.last_registered_time = now
        else:
            ss.pending_raw = raw
            ss.pending_since = now
    else:
        ss.pending_raw = None


def update_presence(hands, now):
    ss = st.session_state
    ss.presence_now = bool(hands)
    if hands:
        ss.last_presence_time = now


def handle_lockout_expiry(now):
    ss = st.session_state
    if ss.lockout_until and now >= ss.lockout_until:
        ss.lockout_until = 0.0
        ss.incorrect_times = []
        ss.status_msg = "Lockout expired. Gesture unlocking re-enabled."


def update_door_and_auto_lock(now, dt_sim):
    ss = st.session_state
    if dt_sim <= 0:
        return

    speed = 70.0 / OPEN_DURATION
    angle = ss.door_angle
    target = ss.door_target

    if angle < target:
        angle = min(target, angle + speed * dt_sim)
    elif angle > target:
        angle = max(target, angle - speed * dt_sim)

    ss.door_angle = angle

    if ss.lock_state == "unlocked":
        no_presence = now - ss.last_presence_time

        if ss.door_target == 70.0 and no_presence >= ss.auto_close_timeout:
            ss.door_target = 0.0
            ss.status_msg = "No presence detected. Auto-closing."

        if no_presence >= ss.hard_lock_timeout and angle > 1.0:
            ss.door_target = 0.0

        if ss.door_target == 0.0 and angle <= 0.7:
            ss.lock_state = "locked"
            ss.open_time = None
            ss.status_msg = "Door closed and re-locked."


def process_demo_actions(now):
    ss = st.session_state
    if not ss.demo_active:
        return

    t = now - ss.demo_start_sim

    for idx, (at, action) in enumerate(DEMO_ACTIONS):
        if t >= at and idx not in ss.demo_actions_done:
            ss.demo_actions_done.add(idx)

            if action == "secure_on":
                ss.secure_mode = True
                ss.status_msg = "Demo: Secure mode toggled ON."
            elif action == "stop":
                ss.demo_active = False
                ss.status_msg = "Guided demo complete. Use manual gestures or restart demo."


# ----------------------------
# Simulation / live input
# ----------------------------
def get_simulation_hands(now):
    ss = st.session_state

    while ss.manual_queue and now >= ss.manual_queue[0][1]:
        ss.manual_queue.popleft()

    if ss.manual_queue:
        pose, end_t, start_t = ss.manual_queue[0]
        local = max(0.0, now - start_t)

        if pose == "crossed":
            left = make_single_hand("open_palm", local, wrist_x=0.47, wrist_y=0.74, scale=0.8, mirror=True)
            right = make_single_hand("open_palm", local, wrist_x=0.53, wrist_y=0.76, scale=0.8, mirror=False)
            return [left, right]

        return [make_single_hand(pose, local)]

    if ss.demo_active:
        t = now - ss.demo_start_sim

        for start, end, pose, hand_count, params in DEMO_POSES:
            if start <= t < end:
                local = t - start

                if pose == "crossed":
                    left = make_single_hand("open_palm", local, wrist_x=0.47, wrist_y=0.74, scale=0.8, mirror=True)
                    right = make_single_hand("open_palm", local, wrist_x=0.53, wrist_y=0.76, scale=0.8, mirror=False)
                    return [left, right]

                return [make_single_hand(pose, local, **params)]

        if t >= DEMO_END:
            ss.demo_active = False
            ss.status_msg = "Guided demo complete. Use manual gestures or restart demo."

        return []

    return []


def get_camera():
    ss = st.session_state

    if ss.source != "Live webcam":
        if ss.cap is not None:
            ss.cap.release()
            ss.cap = None
        return None

    if ss.cap is None:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ss.cap = cap
            ss.cap_error = None
        else:
            ss.cap_error = "Unable to open webcam. Using simulated demo mode."
            ss.cap = None

    return ss.cap


def get_live_hands(bgr_frame):
    ss = st.session_state
    if not MP_AVAILABLE:
        return []

    if ss.mp_hands is None:
        try:
            ss.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            ss.cap_error = f"MediaPipe init failed: {e}"
            return []

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = ss.mp_hands.process(rgb)

    hands = []
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            arr = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=float)
            hands.append(arr)

    return hands


# ----------------------------
# Rendering
# ----------------------------
def draw_transparent_rect(img, pt1, pt2, color, alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def indicator_color(now):
    ss = st.session_state

    if now < ss.flash_until:
        return (0, 0, 255) if int(now * 10) % 2 == 0 else (0, 0, 90)

    if now < ss.lockout_until:
        return (0, 191, 255) if int(now * 4) % 2 == 0 else (0, 90, 140)

    if ss.lock_state == "unlocked":
        return (0, 255, 0)

    return (0, 0, 255)


def draw_door_panel(img, angle_deg):
    L, T, R, B = DOOR_LEFT, DOOR_TOP, DOOR_RIGHT, DOOR_BOTTOM
    width = R - L
    cy = (T + B) // 2

    theta = math.radians(max(0.0, min(80.0, angle_deg)))
    focal = 520.0
    z = width * math.sin(theta)
    factor = focal / (focal + z)

    free_x = L + width * math.cos(theta) * factor
    top_y = cy + (T - cy) * factor
    bot_y = cy + (B - cy) * factor

    pts = np.array([
        [L, T],
        [int(free_x), int(top_y)],
        [int(free_x), int(bot_y)],
        [L, B],
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], (68, 52, 38))
    cv2.polylines(img, [pts], True, (28, 20, 14), 2)

    # Simple inner panel detail
    inset_x = int(L + (free_x - L) * 0.15)
    inset_top = int(T + (top_y - T) * 0.15)
    inset_bot = int(B + (bot_y - B) * 0.15)
    cv2.rectangle(
        img,
        (inset_x, inset_top),
        (int(L + (free_x - L) * 0.85), inset_bot),
        (50, 38, 28),
        2
    )

    # Handle
    hx = int(L + (free_x - L) * 0.86)
    hy = int((top_y + bot_y) / 2)
    cv2.circle(img, (hx, hy), 6, (190, 190, 190), -1)
    cv2.circle(img, (hx, hy), 3, (90, 90, 90), -1)


def draw_hand(img, lm):
    pts = np.column_stack((lm[:, 0] * FRAME_W, lm[:, 1] * FRAME_H)).astype(int)

    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 255), 2)

    for p in pts:
        cv2.circle(img, tuple(p), 2, (0, 255, 0), -1)


def draw_hud(img, now):
    ss = st.session_state

    draw_transparent_rect(img, (8, 8), (270, 142), (0, 0, 0), 0.50)

    lock_text = "UNLOCKED" if ss.lock_state == "unlocked" else "LOCKED"
    lock_color = (0, 255, 0) if ss.lock_state == "unlocked" else (0, 0, 255)
    put_text(img, f"LOCK: {lock_text}", (18, 32), 0.55, lock_color, 1)

    if now < ss.guest_mode_until:
        mode = "GUEST"
        mode_color = (255, 180, 0)
    elif ss.secure_mode:
        mode = "SECURE"
        mode_color = (0, 255, 255)
    else:
        mode = "STANDARD"
        mode_color = (220, 220, 220)

    put_text(img, f"MODE: {mode}", (18, 54), 0.55, mode_color, 1)
    put_text(img, f"DOOR: {int(ss.door_angle)} deg", (18, 76), 0.55, (220, 220, 220), 1)

    pres = "HAND" if ss.presence_now else "NONE"
    no_pres = now - ss.last_presence_time if ss.lock_state == "unlocked" else 0.0
    put_text(img, f"PRESENCE: {pres} {no_pres:.1f}s", (18, 98), 0.55, (220, 220, 220), 1)

    if now < ss.lockout_until:
        put_text(
            img,
            f"COOLDOWN: {int(ss.lockout_until - now)}s",
            (18, 120),
            0.55,
            (0, 191, 255),
            1
        )
    elif now < ss.guest_mode_until:
        put_text(
            img,
            f"GUEST LEFT: {int(ss.guest_mode_until - now)}s",
            (18, 120),
            0.55,
            (255, 180, 0),
            1
        )

    draw_transparent_rect(img, (0, FRAME_H - 44), (FRAME_W, FRAME_H), (0, 0, 0), 0.55)
    status = ss.status_msg
    if len(status) > 82:
        status = status[:79] + "..."
    put_text(img, status, (10, FRAME_H - 16), 0.50, (235, 235, 235), 1)


def render_frame(hands, now, cam_frame=None):
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img[:] = (45, 45, 50)

    # Floor
    cv2.rectangle(img, (0, FRAME_H - 70), (FRAME_W, FRAME_H), (30, 30, 35), -1)

    # Door frame
    cv2.rectangle(img, (DOOR_LEFT - 18, DOOR_TOP - 18), (DOOR_RIGHT + 18, DOOR_BOTTOM + 18), (95, 90, 85), -1)
    cv2.rectangle(img, (DOOR_LEFT - 8, DOOR_TOP - 8), (DOOR_RIGHT + 8, DOOR_BOTTOM + 8), (55, 50, 48), -1)

    # Dark opening
    cv2.rectangle(img, (DOOR_LEFT, DOOR_TOP), (DOOR_RIGHT, DOOR_BOTTOM), (18, 18, 22), -1)

    draw_door_panel(img, st.session_state.door_angle)

    # Frame outline
    cv2.rectangle(img, (DOOR_LEFT - 2, DOOR_TOP - 2), (DOOR_RIGHT + 2, DOOR_BOTTOM + 2), (120, 110, 100), 2)

    # Indicator light
    color = indicator_color(now)
    cv2.circle(img, (DOOR_LEFT + 16, DOOR_TOP + 16), 9, (20, 20, 20), -1)
    cv2.circle(img, (DOOR_LEFT + 16, DOOR_TOP + 16), 6, color, -1)

    # Hands
    for lm in hands:
        draw_hand(img, lm)

    # HUD
    draw_hud(img, now)

    # Live camera thumbnail
    if cam_frame is not None:
        thumb = cv2.resize(cam_frame, (160, 120))
        img[10:130, FRAME_W - 170:FRAME_W - 10] = thumb
        cv2.rectangle(img, (FRAME_W - 172, 8), (FRAME_W - 8, 132), (180, 180, 180), 2)

    return img


def update_ui(now, status_ph, metrics_ph, log_ph):
    ss = st.session_state

    if now < ss.guest_mode_until:
        mode = "GUEST"
    elif ss.secure_mode:
        mode = "SECURE"
    else:
        mode = "STANDARD"

    if now < ss.flash_until:
        ind = "FLASH RED"
    elif now < ss.lockout_until:
        ind = "AMBER COOLDOWN"
    elif ss.lock_state == "unlocked":
        ind = "GREEN"
    else:
        ind = "RED"

    guest_left = max(0.0, ss.guest_mode_until - now)
    cooldown = max(0.0, ss.lockout_until - now)
    no_pres = now - ss.last_presence_time if ss.lock_state == "unlocked" else 0.0
    presence = "HAND" if ss.presence_now else "NONE"

    status_ph.markdown(f"💬 **{ss.status_msg}**")

    metrics_ph.markdown(
        f"""
        | Lock | Mode | Door | Indicator | Presence | Cooldown | Guest |
        |---|---|---|---|---|---|---|
        | {ss.lock_state.upper()} | {mode} | {int(ss.door_angle)}° | {ind} | {presence} ({no_pres:.1f}s) | {cooldown:.0f}s | {guest_left:.0f}s |
        """
    )

    if ss.access_log:
        lines = [
            f"- `{e['time']}` — **{e['method'].upper()}** — {e['event']}"
            for e in ss.access_log
        ]
    else:
        lines = ["- No unlock events yet."]

    log_ph.markdown("#### 📜 Access log (last 5 unlock events)\n" + "\n".join(lines))


# ----------------------------
# App layout
# ----------------------------
init_state()

st.title("🚪 Gesture-Controlled Smart Lock Simulator")
st.caption(
    "Wave unlock, gesture passcode, guest mode, presence auto-lock, failed-attempt lockout, "
    "and access log. Built with Streamlit, OpenCV, MediaPipe Hands, and NumPy. No API key."
)

st.sidebar.header("🎛 Controls")

ui_source = st.sidebar.radio(
    "Input source",
    ["Simulation", "Live webcam"],
    index=0,
    key="ui_source"
)

ui_speed = st.sidebar.slider(
    "Simulation speed",
    0.5,
    4.0,
    2.0,
    0.1,
    key="ui_speed",
    help="Increase to compress the guided demo into under 60 seconds of wall-clock recording."
)

ui_auto_close = st.sidebar.slider(
    "Soft auto-close timeout (s)",
    3.0,
    20.0,
    5.0,
    0.5,
    key="ui_auto_close"
)

ui_hard_lock = st.sidebar.slider(
    "Hard auto-lock timeout (s)",
    5.0,
    30.0,
    8.0,
    0.5,
    key="ui_hard_lock"
)

ui_guest_duration = st.sidebar.slider(
    "Guest mode duration (s)",
    10.0,
    300.0,
    120.0,
    5.0,
    key="ui_guest_duration"
)

ui_wave_min = st.sidebar.slider(
    "Wave zero-crossings required",
    2,
    6,
    3,
    1,
    key="ui_wave_min"
)

c1, c2 = st.sidebar.columns(2)
c1.button("▶️ Start guided demo", on_click=start_guided_demo, use_container_width=True)
c2.button("🔄 Reset", on_click=reset_system, use_container_width=True)

st.sidebar.button("🔐 Toggle secure mode", on_click=toggle_secure_mode, use_container_width=True)
st.sidebar.button("🧑‍‍🧑 Enable guest now", on_click=enable_guest_now, use_container_width=True)

st.sidebar.subheader("🖐 Manual gestures")
m1, m2 = st.sidebar.columns(2)
m1.button("Wave", on_click=lambda: queue_manual("wave", 1.3), use_container_width=True)
m2.button("Static palm", on_click=lambda: queue_manual("open_palm", 0.9), use_container_width=True)
m1.button("Fist", on_click=lambda: queue_manual("fist", 0.9), use_container_width=True)
m2.button("Point", on_click=lambda: queue_manual("point", 0.9), use_container_width=True)
st.sidebar.button("Crossed wrists (guest)", on_click=lambda: queue_manual("crossed", 1.3), use_container_width=True)

pause_label = "▶️ Resume" if st.session_state.paused else "⏸️ Pause"
run_label = "🟢 Start loop" if not st.session_state.running else "🔴 Stop loop"

pc1, pc2 = st.sidebar.columns(2)
pc1.button(pause_label, on_click=toggle_pause, use_container_width=True)
pc2.button(run_label, on_click=toggle_run, use_container_width=True)

warn_ph = st.sidebar.empty()
cam_ph = st.sidebar.empty()

# Sync widget values into session state
ss = st.session_state
ss.source = ui_source
if not MP_AVAILABLE and ss.source == "Live webcam":
    ss.source = "Simulation"
    ss.cap_error = "MediaPipe is not available in this environment. Running in simulation mode."
elif ss.source != "Live webcam":
    ss.cap_error = None

ss.time_scale = float(ui_speed)
ss.auto_close_timeout = float(ui_auto_close)
ss.hard_lock_timeout = max(float(ui_hard_lock), float(ui_auto_close))
ss.guest_duration = float(ui_guest_duration)
ss.wave_min_crossings = int(ui_wave_min)

video_ph = st.empty()
status_ph = st.empty()
metrics_ph = st.empty()
log_ph = st.empty()

# ----------------------------
# Main real-time loop
# ----------------------------
while st.session_state.running:
    loop_start = time.time()
    ss = st.session_state

    now_real = time.time()
    dt_real = min(0.2, now_real - ss.last_loop_real)
    ss.last_loop_real = now_real

    if ss.paused:
        dt_sim = 0.0
    else:
        dt_sim = dt_real * ss.time_scale
        ss.sim_time += dt_sim

    now = ss.sim_time

    hands = []
    cam_frame = None

    try:
        if ss.source == "Live webcam" and MP_AVAILABLE:
            cap = get_camera()
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    cam_frame = cv2.flip(cv2.resize(frame, (FRAME_W, FRAME_H)), 1)
                    hands = get_live_hands(cam_frame)
                else:
                    ss.cap_error = "Webcam frame read failed. Showing simulated scene only."
            else:
                if not ss.cap_error:
                    ss.cap_error = "Webcam unavailable."
        else:
            hands = get_simulation_hands(now)

        if not ss.paused:
            process_demo_actions(now)
            update_presence(hands, now)
            handle_lockout_expiry(now)
            detect_and_handle_gestures(hands, now, dt_sim)
            update_door_and_auto_lock(now, dt_sim)

        rendered = render_frame(hands, now, cam_frame)
        video_ph.image(rendered, channels="BGR", use_container_width=True)

        ss.frame_counter += 1

        if ss.frame_counter % 2 == 0:
            update_ui(now, status_ph, metrics_ph, log_ph)

            if ss.cap_error:
                warn_ph.warning(ss.cap_error)
            else:
                warn_ph.empty()

            if cam_frame is not None:
                cam_ph.image(
                    cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )
            else:
                cam_ph.empty()

    except Exception as e:
        st.error(f"Runtime error: {e}")
        ss.running = False
        break

    elapsed = time.time() - loop_start
    time.sleep(max(0.001, 0.05 - elapsed))

if not st.session_state.running:
    st.info("Simulation loop stopped. Use **Start loop** in the sidebar to resume.")