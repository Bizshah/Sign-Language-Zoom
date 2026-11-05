import cv2
import numpy as np
import pyvirtualcam

WIDTH = 1280
HEIGHT = 720
FPS = 30

# Connect to laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open laptop webcam!")
    exit()

print("Laptop webcam connected!")

try:
    with pyvirtualcam.Camera(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        fmt=pyvirtualcam.PixelFormat.BGR
    ) as cam:

        print("Virtual camera started:", cam.device)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame!")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            cam.send(frame)
            cam.sleep_until_next_frame()

except KeyboardInterrupt:
    print("Camera stopped.")

finally:
    cap.release()