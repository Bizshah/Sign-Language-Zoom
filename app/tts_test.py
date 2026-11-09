import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
import threading
import queue

from collections import deque


# ---------------- SETTINGS ----------------

WIDTH = 1280
HEIGHT = 720
FPS = 30

BUFFER_SECONDS = 2
REST_SECONDS = 1.0
MOVEMENT_THRESHOLD = 0.008

MODEL_PATH = "hand_landmarker.task"

frame_buffer = deque()
previous_positions = []

segment_active = False
last_movement_time = 0

speech_queue = queue.Queue()


# ---------------- SPEECH WORKER ----------------

def speech_worker():
    while True:
        text = speech_queue.get()

        if text is None:
            speech_queue.task_done()
            break

        powershell_command = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{text}')"
        )

        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", powershell_command],
                check=True
            )
        except Exception as error:
            print("Speech error:", error)

        speech_queue.task_done()


# ---------------- HAND DETECTION ----------------

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

    current_positions = []
    raised = False
    moving = False

    for hand in result.hand_landmarks:
        wrist = hand[0]

        current = np.array([wrist.x, wrist.y])
        current_positions.append(current)

        # Rough raised-hand check
        if wrist.y < 0.75:
            raised = True

    for current, previous in zip(
        current_positions, previous_positions
    ):
        distance = np.linalg.norm(current - previous)

        if distance > MOVEMENT_THRESHOLD:
            moving = True

    previous_positions = current_positions

    return raised, moving


# ---------------- MAIN CAMERA LOOP ----------------

def main():
    global segment_active, last_movement_time

    speech_thread = threading.Thread(
        target=speech_worker,
        daemon=True
    )
    speech_thread.start()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    if not camera.isOpened():
        print("Could not open webcam.")
        speech_queue.put(None)
        return

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    start_time = time.monotonic()

    print("SignBridge realtime camera started.")
    print("Press Q to stop.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                now = time.monotonic()

                # Keep the last two seconds of frames
                frame_buffer.append((now, frame.copy()))

                while (
                    frame_buffer
                    and now - frame_buffer[0][0] > BUFFER_SECONDS
                ):
                    frame_buffer.popleft()

                # MediaPipe detection
                small_frame = cv2.resize(frame, (640, 360))

                 rgb_frame = cv2.cvtColor(
                  small_frame, cv2.COLOR_BGR2RGB
)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    (now - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                raised, moving = detect_hand_and_movement(
                    result
                )

                # Start a sign segment
                if raised and moving:
                    last_movement_time = now

                    if not segment_active:
                        segment_active = True
                        print("SIGN SEGMENT STARTED")

                # End a sign segment after resting
                if (
                    segment_active
                    and now - last_movement_time >= REST_SECONDS
                ):
                    segment_active = False

                    print("SIGN SEGMENT ENDED")
                    print(predict(frame_buffer))
                    speech_queue.put("Sign captured.")

                    # Speech happens in the background
                    speech_queue.put("Sign captured.")

                # Draw detected hand landmarks
                if result.hand_landmarks:
                    for hand in result.hand_landmarks:

                        points = []

                        for landmark in hand:
                            x = int(landmark.x * WIDTH)
                            y = int(landmark.y * HEIGHT)
                            points.append((x, y))

                        for start, end in HAND_CONNECTIONS:
                            cv2.line(
                                frame,
                                points[start],
                                points[end],
                                (0, 255, 0),
                                2
                            )

                        for point in points:
                            cv2.circle(
                                frame,
                                point,
                                4,
                                (0, 0, 255),
                                -1
                            )

                    status = "HAND DETECTED"
                else:
                    status = "NO HAND"

                cv2.putText(
                    frame,
                    status,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "SignBridge - Realtime",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        speech_queue.put(None)

        print("Webcam released.")


if __name__ == "__main__":
    main()