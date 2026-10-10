"""Write a single-frame MJPEG that Chrome can use as a synthetic camera.

Usage: .venv\\Scripts\\python.exe tests\\make_fake_camera.py
Writes C:\\Users\\Public\\aquascan_fakecam.mjpeg (Chrome's --use-file-for-fake-video-capture
silently falls back to the real camera when the path contains spaces).
The source is a dataset test image with a clear bottle (detects at ~92%).
This verifies software rendering only; it says nothing about physical-bottle detection.
"""
import cv2

SRC = "dataset_clean/floating_waste_v10/test/images/frame_000041_jpg.rf.b9ae17d3d0812e292d8afd19bc69d256.jpg"
OUT = r"C:\Users\Public\aquascan_fakecam.mjpeg"

img = cv2.resize(cv2.imread(SRC), (640, 480))
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
open(OUT, "wb").write(buf.tobytes())
print("wrote", OUT)
