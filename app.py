import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from flask import Flask, Response, jsonify
from collections import deque
import math

app = Flask(__name__)

# ──────────────────────────────────────────────
#  MediaPipe setup
# ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,        # enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ──────────────────────────────────────────────
#  Landmark indices
# ──────────────────────────────────────────────
# Iris centres (MediaPipe refine_landmarks indices)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corners
LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# Head-pose reference points (6-point model)
HEAD_POSE_PTS = [1, 152, 263, 33, 287, 57]   # nose, chin, L-eye, R-eye, L-mouth, R-mouth

# 3-D model points (generic head, mm)
MODEL_3D = np.array([
    [0.0,    0.0,    0.0  ],   # nose tip
    [0.0,   -330.0, -65.0 ],   # chin
    [-225.0,  170.0,-135.0],   # left eye corner
    [ 225.0,  170.0,-135.0],   # right eye corner
    [-150.0, -150.0,-125.0],   # left mouth
    [ 150.0, -150.0,-125.0],   # right mouth
], dtype=np.float64)

# ──────────────────────────────────────────────
#  Smoothing buffers
# ──────────────────────────────────────────────
SMOOTH = 12
dx_buf    = deque(maxlen=SMOOTH)
dy_buf    = deque(maxlen=SMOOTH)
pitch_buf = deque(maxlen=SMOOTH)
yaw_buf   = deque(maxlen=SMOOTH)

# ──────────────────────────────────────────────
#  Thresholds
# ──────────────────────────────────────────────
THRESH = dict(
    phone_pitch   =  8.0,   # head pitched down  (looking at phone)
    phone_dy      =  0.08,  # iris looking down
    side_yaw      = 15.0,   # head turned sideways (secondary screen)
    side_dx       =  0.09,  # iris looking sideways
    up_pitch      = -6.0,   # head tilted back (distraction up)
    alert_secs    =  2.5,   # sustained time before alert
)

# ──────────────────────────────────────────────
#  Shared state (thread-safe via lock)
# ──────────────────────────────────────────────
state_lock = threading.Lock()
gaze_state = {
    "status":        "initializing",
    "alert":         False,
    "dx":            0.0,
    "dy":            0.0,
    "pitch":         0.0,
    "yaw":           0.0,
    "gaze_h":        0.0,
    "gaze_v":        0.0,
    "alert_elapsed": 0.0,
    "fps":           0.0,
}

frame_lock    = threading.Lock()
latest_frame  = None
camera_active = False

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def iris_displacement(landmarks, eye_corners, iris_pts, w, h):
    """Return normalised (dx, dy) of iris relative to eye bounding box."""
    lx = int(landmarks[eye_corners[0]].x * w)
    ly = int(landmarks[eye_corners[0]].y * h)
    rx = int(landmarks[eye_corners[1]].x * w)
    ry = int(landmarks[eye_corners[1]].y * h)

    cx = np.mean([landmarks[i].x * w for i in iris_pts])
    cy = np.mean([landmarks[i].y * h for i in iris_pts])

    eye_w = math.dist((lx, ly), (rx, ry)) + 1e-6
    mid_x = (lx + rx) / 2
    mid_y = (ly + ry) / 2

    return (cx - mid_x) / eye_w, (cy - mid_y) / eye_w


def head_pose(landmarks, w, h):
    """Return (pitch, yaw) in degrees via solvePnP."""
    img_pts = np.array([
        [landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_POSE_PTS
    ], dtype=np.float64)

    focal  = w
    centre = (w / 2, h / 2)
    cam    = np.array([[focal, 0, centre[0]],
                       [0, focal, centre[1]],
                       [0, 0, 1]], dtype=np.float64)
    dist   = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    pitch = math.degrees(math.atan2(-rmat[2,0], sy))
    yaw   = math.degrees(math.atan2( rmat[1,0], rmat[0,0]))
    return pitch, yaw


def smooth(buf, val):
    buf.append(val)
    return float(np.mean(buf))


def classify(pitch, yaw, gaze_h, gaze_v):
    """Return (status_str, is_alert_condition)."""
    if (pitch > THRESH["phone_pitch"] or gaze_v > THRESH["phone_dy"]):
        return "looking_down", True
    if (abs(yaw) > THRESH["side_yaw"] or abs(gaze_h) > THRESH["side_dx"]):
        side = "right" if (yaw > 0 or gaze_h > 0) else "left"
        return f"looking_sideways_{side}", True
    if pitch < THRESH["up_pitch"]:
        return "looking_up", True
    return "focused", False


# ──────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────
STATUS_COLORS = {
    "focused":              (0, 220, 80),
    "looking_down":         (0, 140, 255),
    "looking_sideways_left":(255, 180,  0),
    "looking_sideways_right":(255, 180, 0),
    "looking_up":           (80,  80, 255),
    "initializing":         (180, 180, 180),
}
STATUS_LABELS = {
    "focused":               "FOCUSED",
    "looking_down":          "PHONE DETECTED",
    "looking_sideways_left": "SECONDARY SCREEN",
    "looking_sideways_right":"SECONDARY SCREEN",
    "looking_up":            "DISTRACTED (UP)",
    "initializing":          "INITIALIZING…",
}

def draw_hud(frame, st):
    h, w = frame.shape[:2]
    status  = st["status"]
    color   = STATUS_COLORS.get(status, (180, 180, 180))
    label   = STATUS_LABELS.get(status, status.upper())
    alert   = st["alert"]
    elapsed = st["alert_elapsed"]

    # ── semi-transparent top bar ──────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # status pill
    pill_x, pill_y = 12, 10
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)[0]
    cv2.rectangle(frame, (pill_x - 6, pill_y - 4),
                  (pill_x + text_size[0] + 6, pill_y + text_size[1] + 6),
                  color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (pill_x, pill_y + text_size[1]),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (10, 10, 20), 1, cv2.LINE_AA)

    # FPS
    fps_txt = f"{st['fps']:.0f} FPS"
    cv2.putText(frame, fps_txt, (w - 90, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # ── metric panel bottom-left ──────────────────
    panel_x, panel_y = 12, h - 130
    metrics = [
        (f"dx  {st['dx']:+.3f}", (200, 200, 200)),
        (f"dy  {st['dy']:+.3f}", (200, 200, 200)),
        (f"pitch {st['pitch']:+.1f}°", (160, 200, 255)),
        (f"yaw   {st['yaw']:+.1f}°", (160, 200, 255)),
    ]
    for i, (txt, c) in enumerate(metrics):
        cv2.putText(frame, txt, (panel_x, panel_y + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 1, cv2.LINE_AA)

    # ── mini gaze compass ─────────────────────────
    cx, cy, r = w - 58, h - 58, 42
    cv2.circle(frame, (cx, cy), r, (40, 40, 50), -1)
    cv2.circle(frame, (cx, cy), r, (80, 80, 90), 1)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), (60, 60, 70), 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), (60, 60, 70), 1)
    gx = int(cx + np.clip(st["gaze_h"], -1, 1) * (r - 8))
    gy = int(cy + np.clip(st["gaze_v"], -1, 1) * (r - 8))
    cv2.circle(frame, (gx, gy), 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (200, 200, 200), -1)

    # ── alert flash border ────────────────────────
    if alert:
        prog = min(elapsed / THRESH["alert_secs"], 1.0)
        thick = max(2, int(6 * prog))
        flash = (elapsed % 0.6) < 0.3
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thick)
        # countdown bar
        bar_w = int(w * prog)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), color, -1)

    return frame


# ──────────────────────────────────────────────
#  Camera loop (runs in background thread)
# ──────────────────────────────────────────────
def camera_loop():
    global latest_frame, camera_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with state_lock:
            gaze_state["status"] = "no_camera"
        return

    camera_active = True
    prev_time     = time.time()
    alert_start   = None

    while camera_active:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = face_mesh.process(rgb)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        local = dict(gaze_state)   # snapshot

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # iris displacement (average both eyes)
            ldx, ldy = iris_displacement(lm, LEFT_EYE_CORNERS,  LEFT_IRIS,  w, h)
            rdx, rdy = iris_displacement(lm, RIGHT_EYE_CORNERS, RIGHT_IRIS, w, h)
            raw_dx = (ldx + rdx) / 2
            raw_dy = (ldy + rdy) / 2

            # head pose
            raw_pitch, raw_yaw = head_pose(lm, w, h)

            # smooth
            sdx   = smooth(dx_buf,    raw_dx)
            sdy   = smooth(dy_buf,    raw_dy)
            spitch= smooth(pitch_buf, raw_pitch)
            syaw  = smooth(yaw_buf,   raw_yaw)

            # combined gaze (weighted sum)
            gaze_h = sdx * 0.5 + math.sin(math.radians(syaw))  * 0.5
            gaze_v = sdy * 0.5 + math.sin(math.radians(spitch)) * 0.5

            status, is_distracted = classify(spitch, syaw, sdx, sdy)

            if is_distracted:
                if alert_start is None:
                    alert_start = now
                elapsed = now - alert_start
                alert   = elapsed >= THRESH["alert_secs"]
            else:
                alert_start = None
                elapsed = 0.0
                alert   = False

            local.update(dict(
                status        = status,
                alert         = alert,
                dx            = round(sdx, 4),
                dy            = round(sdy, 4),
                pitch         = round(spitch, 2),
                yaw           = round(syaw,   2),
                gaze_h        = round(gaze_h,  4),
                gaze_v        = round(gaze_v,  4),
                alert_elapsed = round(elapsed, 2),
                fps           = round(fps, 1),
            ))
        else:
            local["status"] = "no_face"
            local["fps"]    = round(fps, 1)
            alert_start     = None

        with state_lock:
            gaze_state.update(local)

        annotated = draw_hud(frame, local)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buf.tobytes()

    cap.release()


# ──────────────────────────────────────────────
#  Flask routes
# ──────────────────────────────────────────────
def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        return jsonify(dict(gaze_state))


@app.route("/")
def index():
    return open("index.html").read()


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5001, debug=False)
