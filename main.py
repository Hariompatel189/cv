import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime
import csv
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_NAME   = "Gym Rep Counter – Pose Estimation"
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
LOG_FILE      = "workout_log.csv"

# Visibility threshold for landmarks
VIS_THRESH = 0.6

# Exercise angle thresholds  (down_angle, up_angle)
EXERCISE_CONFIG = {
    "Bicep Curl":    {"joints": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),   "down": 160, "up": 40},
    "Squat":         {"joints": ("LEFT_HIP",      "LEFT_KNEE",  "LEFT_ANKLE"),   "down": 160, "up": 90},
    "Push-Up":       {"joints": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),   "down": 90,  "up": 160},
    "Shoulder Press":{"joints": ("LEFT_ELBOW",    "LEFT_SHOULDER","LEFT_HIP"),   "down": 70,  "up": 160},
    "Lateral Raise": {"joints": ("LEFT_HIP",      "LEFT_SHOULDER","LEFT_ELBOW"), "down": 20,  "up": 90},
}

EXERCISES   = list(EXERCISE_CONFIG.keys())
COLORS = {
    "green":  (0, 220, 80),
    "red":    (0, 60, 220),
    "yellow": (0, 200, 255),
    "dark":   (30, 30, 30),
    "cyan":   (255, 220, 0),
    "white":  (255, 255, 255),
}

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_landmark_coords(landmarks, name, w, h):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return (int(lm.x * w), int(lm.y * h)), lm.visibility


def calculate_angle(a, b, c):
    """Return angle at point b formed by a-b-c (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle  = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(angle, 1)


def draw_angle_arc(frame, vertex, angle, color, radius=40):
    cv2.putText(frame, f"{int(angle)}", (vertex[0] - 20, vertex[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_progress_bar(frame, x, y, w, h, pct, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), cv2.FILLED)
    filled = int(w * pct)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS["white"], 1)


def draw_top_bar(frame, exercise, elapsed):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), COLORS["dark"], cv2.FILLED)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"Exercise: {exercise}   |   {ts}   |   Session: {elapsed}s",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["cyan"], 1)


def draw_bottom_bar(frame, tips):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), COLORS["dark"], cv2.FILLED)
    cv2.putText(frame, tips, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def draw_rep_panel(frame, reps, stage, goal):
    """Draw translucent rep counter panel (top-left)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 60), (200, 200), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "REPS", (30, 95),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COLORS["yellow"], 1)
    cv2.putText(frame, str(reps), (50, 165),
                cv2.FONT_HERSHEY_DUPLEX, 2.8, COLORS["green"], 3)

    stage_color = COLORS["green"] if stage == "UP" else COLORS["red"]
    cv2.putText(frame, stage if stage else "---", (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, stage_color, 2)

    # Goal progress bar
    pct = min(reps / goal, 1.0) if goal else 0
    draw_progress_bar(frame, 10, 205, 190, 12, pct, COLORS["green"])
    cv2.putText(frame, f"Goal: {goal}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def draw_exercise_selector(frame, exercises, current_idx):
    """Draw exercise selection panel (top-right)."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 240, 30 * len(exercises) + 20
    x0, y0 = w - panel_w - 10, 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "EXERCISES  [1-5]", (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["cyan"], 1)

    for i, ex in enumerate(exercises):
        yt = y0 + 40 + i * 28
        color = COLORS["green"] if i == current_idx else (160, 160, 160)
        prefix = "▶ " if i == current_idx else f"{i+1}. "
        cv2.putText(frame, prefix + ex, (x0 + 8, yt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ─────────────────────────────────────────────
#  WORKOUT LOG
# ─────────────────────────────────────────────
def log_set(exercise, reps, duration_s):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Date", "Time", "Exercise", "Reps", "Duration(s)"])
        now = datetime.now()
        writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                         exercise, reps, duration_s])
    print(f"[LOG] Set saved → {exercise}: {reps} reps in {duration_s}s")


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    # State
    current_exercise = 0
    rep_count   = 0
    stage       = None          # "UP" or "DOWN"
    goal_reps   = 10
    set_start   = time.time()
    session_start = time.time()
    form_ok     = True

    print("[INFO] Starting Gym Rep Counter…")
    print("       Keys: 1-5 → Switch exercise | R → Reset reps | Q → Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        elapsed = int(time.time() - session_start)
        ex_name = EXERCISES[current_exercise]
        cfg     = EXERCISE_CONFIG[ex_name]
        j1, j2, j3 = cfg["joints"]

        angle = None

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=COLORS["yellow"], thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 200, 200), thickness=2),
            )

            # Get joint coordinates
            p1, v1 = get_landmark_coords(lms, j1, w, h)
            p2, v2 = get_landmark_coords(lms, j2, w, h)
            p3, v3 = get_landmark_coords(lms, j3, w, h)

            # Only count when joints are clearly visible
            if min(v1, v2, v3) > VIS_THRESH:
                angle = calculate_angle(p1, p2, p3)
                draw_angle_arc(frame, p2, angle, COLORS["cyan"])

                # Draw joint circles
                for pt in [p1, p2, p3]:
                    cv2.circle(frame, pt, 8, COLORS["cyan"], -1)
                    cv2.circle(frame, pt, 8, COLORS["white"], 2)

                # Rep counting logic
                if angle > cfg["down"]:
                    stage = "DOWN"
                if angle < cfg["up"] and stage == "DOWN":
                    stage = "UP"
                    rep_count += 1
                    print(f"[REP] {ex_name} → {rep_count}")

                    if rep_count >= goal_reps:
                        duration = int(time.time() - set_start)
                        log_set(ex_name, rep_count, duration)

                # Angle feedback bar (right side)
                down_a, up_a = cfg["down"], cfg["up"]
                norm = 1.0 - np.clip((angle - up_a) / (down_a - up_a + 1e-6), 0, 1)
                bar_x, bar_y, bar_h_px = w - 50, 80, h - 130
                draw_progress_bar(frame, bar_x, bar_y, 20, bar_h_px,
                                  norm, COLORS["green"])
                cv2.putText(frame, f"{int(angle)}°", (bar_x - 5, bar_y + bar_h_px + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)

            else:
                form_ok = False
                cv2.putText(frame, "⚠ Move closer / adjust camera",
                            (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["red"], 2)

        # ── UI Panels ──
        draw_top_bar(frame, ex_name, elapsed)
        draw_rep_panel(frame, rep_count, stage, goal_reps)
        draw_exercise_selector(frame, EXERCISES, current_exercise)
        draw_bottom_bar(frame,
            "Keys: [1-5] Switch Exercise  |  [R] Reset Reps  |  [G] Set Goal  |  [Q] Quit")

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # ── Key Bindings ──
        if key == ord('q'):
            print("[INFO] Quitting…")
            break
        elif key == ord('r'):
            duration = int(time.time() - set_start)
            if rep_count > 0:
                log_set(ex_name, rep_count, duration)
            rep_count = 0
            stage     = None
            set_start = time.time()
            print("[INFO] Reps reset.")
        elif key == ord('g'):
            goal_reps = goal_reps + 5 if goal_reps < 30 else 5
            print(f"[INFO] Goal set to {goal_reps} reps")
        elif ord('1') <= key <= ord('5'):
            idx = key - ord('1')
            if idx < len(EXERCISES):
                if rep_count > 0:
                    log_set(ex_name, rep_count, int(time.time() - set_start))
                current_exercise = idx
                rep_count = 0
                stage     = None
                set_start = time.time()
                print(f"[INFO] Switched to: {EXERCISES[current_exercise]}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"[INFO] Session ended. Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
