"""Backend webcam soak: N consecutive laptop_webcam /api/predict requests.

Usage: .venv\\Scripts\\python.exe tests\\soak_webcam.py [image] [count]
Requires the backend running on http://127.0.0.1:8000.
Pass = every response HTTP 200 with valid JSON echoing the frame_id.
"""
import sys
import time
import requests

image = sys.argv[1] if len(sys.argv) > 1 else "scratch_frame.jpg"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
frame = open(image, "rb").read()

codes, lat, bad_json = {}, [], 0
for i in range(count):
    t = time.time()
    r = requests.post(
        "http://127.0.0.1:8000/api/predict",
        files={"file": ("f.jpg", frame, "image/jpeg")},
        data={"conf_threshold": 0.25, "source": "laptop_webcam", "frame_id": i + 1},
        timeout=10,
    )
    lat.append((time.time() - t) * 1000)
    codes[r.status_code] = codes.get(r.status_code, 0) + 1
    try:
        j = r.json()
        assert "detections" in j and j.get("hardware_metadata", {}).get("frame_id") == i + 1
    except Exception:
        bad_json += 1

lat.sort()
ok = codes.get(200, 0) == count and bad_json == 0
print(f"status codes: {codes} | bad JSON: {bad_json} | "
      f"latency ms p50={lat[len(lat) // 2]:.0f} p95={lat[int(len(lat) * 0.95)]:.0f} max={lat[-1]:.0f}")
print(f"[{'PASSED' if ok else 'FAILED'}] {codes.get(200, 0)}/{count} HTTP 200")
sys.exit(0 if ok else 1)
