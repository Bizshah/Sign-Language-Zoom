"""
ml/webcam_landmarks_tasks.py

Live hand + pose + face landmarks from the webcam using the MediaPipe
Tasks API (works on newer MediaPipe versions where mp.solutions is gone).
Shows FPS and CPU load on screen.

The three .task model files are downloaded automatically into ml/models/
the first time you run it.

Controls:
    q  -> quit
    f  -> toggle face landmarks on/off (face is the heaviest part)
"""

import time
import urllib.request
from collections import deque
from pathlib import Path


import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import psutil  # optional: python -m pip install psutil
except ImportError:
    psutil = None

# ---------------- Settings ----------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DRAW_FACE = True
FPS_WINDOW = 30
MODEL_DIR = Path(__file__).parent / "models"

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
}
# ------------------------------------------

# Skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def ensure_models():
    """Download the .task model files if they are not already on disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = MODEL_DIR / name
        if not path.exists():
            print(f"Downloading {name} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                if path.exists():
                    path.unlink()
                raise RuntimeError(
                    f"Could not download {name}.\n"
                    f"Download it manually from:\n  {url}\n"
                    f"and save it to:\n  {path}"
                ) from e
    return MODEL_DIR


def create_landmarkers():
    base = lambda name: mp_python.BaseOptions(
        model_asset_path=str(MODEL_DIR / name)
    )
    video = vision.RunningMode.VIDEO

    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base("hand_landmarker.task"),
            running_mode=video,
            num_hands=2,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base("pose_landmarker_lite.task"),
            running_mode=video,
        )
    )
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base("face_landmarker.task"),
            running_mode=video,
            num_faces=1,
        )
    )
    return hands, pose, face


def draw_points(frame, landmarks, connections=None,
                line_color=(255, 255, 255), point_color=(0, 255, 0), radius=3):
    """Draw one set of normalized landmarks (x, y in 0-1) on the frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    if connections:
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], line_color, 2)
    for p in pts:
        cv2.circle(frame, p, radius, point_color, -1)


def main():
    ensure_models()
    hands, pose, face = create_landmarkers()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. Try another CAMERA_INDEX."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    draw_face = DRAW_FACE
    frame_times = deque(maxlen=FPS_WINDOW)
    prev_time = time.time()
    last_ts = -1
    fps, cpu = 0.0, 0.0

    if psutil:
        psutil.cpu_percent(interval=None)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror view

            # BGR -> RGB for MediaPipe
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamps must be strictly increasing (milliseconds)
            ts = int(time.monotonic() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            # Run the models
            hand_res = hands.detect_for_video(mp_image, ts)
            pose_res = pose.detect_for_video(mp_image, ts)
            face_res = face.detect_for_video(mp_image, ts) if draw_face else None

            # Draw
            if face_res and face_res.face_landmarks:
                for lms in face_res.face_landmarks:
                    draw_points(frame, lms, None, point_color=(200, 200, 0), radius=1)

            for lms in pose_res.pose_landmarks:
                draw_points(frame, lms, POSE_CONNECTIONS,
                            line_color=(255, 255, 255), point_color=(0, 0, 255))

            for lms in hand_res.hand_landmarks:
                draw_points(frame, lms, HAND_CONNECTIONS,
                            line_color=(0, 255, 255), point_color=(0, 255, 0))

            # FPS (averaged over last N frames)
            now = time.time()
            frame_times.append(now - prev_time)
            prev_time = now
            avg_dt = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            if psutil:
                cpu = psutil.cpu_percent(interval=None)

            color = (0, 255, 0) if fps >= 20 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if psutil:
                cv2.putText(frame, f"CPU: {cpu:.0f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Face: {'ON' if draw_face else 'OFF'} (press f)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Webcam Landmarks (Tasks API)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                draw_face = not draw_face

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pose.close()
        face.close()
        print(f"Last FPS: {fps:.1f}" + (f" | Last CPU: {cpu:.0f}%" if psutil else ""))


if __name__ == "__main__":
    main()