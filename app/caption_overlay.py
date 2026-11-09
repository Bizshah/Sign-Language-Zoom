import cv2
import pyvirtualcam
import numpy as np
import time

WIDTH = 1280
HEIGHT = 720
FPS = 30

CAPTION = "Hello (test caption)"

def draw_caption(frame, text, alpha):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()
        size = cv2.getTextSize(
            test_line, font, font_scale, thickness
        )[0]

        if size[0] < WIDTH - 100:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    bar_height = 60 + len(lines) * 45
    y1 = HEIGHT - bar_height

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, y1), (WIDTH, HEIGHT), (0, 0, 0), -1
    )

    cv2.addWeighted(
        overlay, alpha, frame, 1 - alpha, 0, frame
    )

    y = y1 + 55

    for line in lines:
        cv2.putText(
            frame, line, (50, y),
            font, font_scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )
        y += 45

    return frame


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("Starting SignBridge caption camera...")

try:
    with pyvirtualcam.Camera(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS
    ) as virtual_camera:

        start_time = time.time()

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            elapsed = time.time() - start_time

            if elapsed < 8:
                alpha = 0.85
            elif elapsed < 10:
                alpha = 0.85 * (10 - elapsed) / 2
            else:
                alpha = 0

            if alpha > 0:
                frame = draw_caption(frame, CAPTION, alpha)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            virtual_camera.send(rgb_frame)
            virtual_camera.sleep_until_next_frame()

except KeyboardInterrupt:
    print("Camera stopped.")

finally:
    camera.release()
    print("Webcam released.")