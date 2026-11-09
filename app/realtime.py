import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

WIDTH = 1280
HEIGHT = 720
FPS = 30
BUFFER_SECONDS = 2
REST_SECONDS = 1.0
MOVEMENT_THRESHOLD = 0.025

frame_buffer = deque()
segment_active = False
last_movement_time = 0
previous_positions = []

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def predict(buffer):
    return "Prediction not implemented yet"


def detect_hand_and_movement(result):
    global previous_positions

    if not result.hand_landmarks:
        previous_positions = []
        return False, False

    positions = []
    raised = False
    moving = False

    for index, hand in enumerate(result.hand_landmarks):
        wrist = hand[0]
        current = np.array([wrist.x, wrist.y])
        positions.append(current)

        if wrist.y < 0.75:
            raised = True

        if index < len(previous_positions):
            distance = np.linalg.norm(
                current - previous_positions[index]
            )
            if distance > MOVEMENT_THRESHOLD:
                moving = True

    previous_positions = positions
    return raised, moving


def main():
    global segment_active, last_movement_time

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("SignBridge realtime detection started.")
    print("Press Q to stop.")

    start_time = time.monotonic()

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                now = time.monotonic()

                frame_buffer.append((now, frame.copy()))

                while (frame_buffer and
                       now - frame_buffer[0][0] > BUFFER_SECONDS):
                    frame_buffer.popleft()

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int((now - start_time) * 1000)

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                raised, moving = detect_hand_and_movement(result)

                if raised and moving:
                    last_movement_time = now

                    if not segment_active:
                        segment_active = True
                        print("SIGN SEGMENT STARTED")

                if (segment_active and
                        now - last_movement_time > REST_SECONDS):
                    segment_active = False
                    print("SIGN SEGMENT ENDED")
                    print("Prediction:", predict(list(frame_buffer)))

                # Draw hand landmarks
                if result.hand_landmarks:
                    for hand in result.hand_landmarks:
                        points = [
                            (int(lm.x * WIDTH),
                             int(lm.y * HEIGHT))
                            for lm in hand
                        ]

                        for start, end in HAND_CONNECTIONS:
                            cv2.line(
                                frame, points[start], points[end],
                                (0, 255, 0), 2
                            )

                        for point in points:
                            cv2.circle(
                                frame, point, 5, (0, 0, 255), -1
                            )

                    status = "HAND DETECTED"
                    color = (0, 255, 0)
                else:
                    status = "NO HAND"
                    color = (0, 0, 255)

                cv2.putText(
                    frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2
                )

                cv2.imshow("SignBridge - Realtime", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released.")


if __name__ == "__main__":
    main()