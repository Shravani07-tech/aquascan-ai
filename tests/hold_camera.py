"""Hold the laptop webcam with OpenCV for N seconds (simulates another app owning the camera).

Usage: .venv\\Scripts\\python.exe tests\\hold_camera.py [seconds]
Used by the Camera busy / Try Again browser test (tests/browser/occupied.js).
"""
import sys
import time
import cv2

cap = cv2.VideoCapture(0)
ok, _ = cap.read()
print("OPENCV_HOLDING", ok, flush=True)
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 12)
cap.release()
print("RELEASED", flush=True)
