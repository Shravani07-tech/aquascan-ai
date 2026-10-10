"""Regression test #18: POST /api/edge/ingest (edge mode: the Pi detects, the dashboard displays).

Usage: .venv\\Scripts\\python.exe tests\\test_edge_ingest.py   (in-process FastAPI TestClient; no server needed)
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.inference import inference_service

client = TestClient(app)
FRAME = open("scratch_frame.jpg", "rb").read()
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


def ingest(meta, data=FRAME, filename="frame.jpg"):
    m = meta if isinstance(meta, str) else json.dumps(meta)
    return client.post("/api/edge/ingest", files={"file": (filename, data, "image/jpeg")}, data={"meta": m})


META = {"device_id": "pi-test", "camera_id": "cam-test", "frame_id": 7, "captured_at": 1.0e9, "imgsz": 416,
        "conf": 0.25, "inference_ms": 105.5, "fps": 9.04, "cpu_temp_c": 61.2,
        "detections": [{"class": "bottle", "confidence": 0.924, "bbox": [410.6, 141.1, 553.6, 242.6]},
                       {"class": "plastic", "confidence": 0.815, "bbox": [271.1, 182.3, 414.8, 257.2]}]}

print("=== #18 /api/edge/ingest ===")

# 1. Valid ingest, and no laptop inference on this path
real_predict = inference_service.predict
calls = []
inference_service.predict = lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("inference called"))
try:
    r = ingest(META)
finally:
    inference_service.predict = real_predict
check("valid ingest -> 200", r.status_code == 200, r.text)
check("no laptop inference during ingest", not calls)
check("response echoes frame_id and count", r.json().get("frame_id") == 7 and r.json().get("count") == 2, r.text)

s = client.get("/api/rpi/latest").json()
meta, pred = s.get("hardware_metadata") or {}, s.get("prediction") or {}
check("/api/rpi/latest active", s.get("active") is True, json.dumps(s)[:200])
check("age_seconds reported", isinstance(s.get("age_seconds"), (int, float)) and s["age_seconds"] < 5)
check("metadata marks edge inference", meta.get("inference_location") == "edge" and meta.get("source") == "raspberry_pi_edge")
check("Pi telemetry stored", meta.get("edge_fps") == 9.04 and meta.get("inference_ms") == 105.5 and meta.get("cpu_temp_c") == 61.2
      and meta.get("device_id") == "pi-test" and meta.get("camera_id") == "cam-test" and meta.get("frame_id") == 7)
check("detections stored in dashboard format",
      pred.get("count") == 2 and [d["class"] for d in pred["detections"]] == ["bottle", "plastic"]
      and pred["detections"][0]["bbox"] == [410.6, 141.1, 553.6, 242.6] and pred["detections"][0]["color"] == "#00f5d4")
check("category counts", pred.get("category_counts", {}).get("bottle") == 1 and pred["category_counts"].get("plastic") == 1)
check("stats from the Pi (latency_ms)", pred.get("latency_ms") == {"total": 105.5, "fps": 9.04})
check("image dimensions from the frame", pred.get("image_dimensions") == {"width": 640, "height": 480})
img = client.get("/api/rpi/latest/image")
check("frame served back byte-identical", img.status_code == 200 and img.content == FRAME)

# 2. Zero detections is valid
r = ingest({**META, "frame_id": 8, "detections": []})
check("empty detections -> 200, count 0", r.status_code == 200 and client.get("/api/rpi/latest").json()["prediction"]["count"] == 0)

# 3. Invalid payloads -> 400 (and do not replace the stored frame)
bad = [
    ("meta not JSON", lambda: ingest("{not json")),
    ("meta not an object", lambda: ingest("[1, 2]")),
    ("unknown class", lambda: ingest({**META, "detections": [{"class": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]})),
    ("confidence > 1", lambda: ingest({**META, "detections": [{"class": "bottle", "confidence": 1.5, "bbox": [0, 0, 1, 1]}]})),
    ("bbox wrong length", lambda: ingest({**META, "detections": [{"class": "bottle", "confidence": 0.5, "bbox": [0, 0, 1]}]})),
    ("bbox non-numeric", lambda: ingest({**META, "detections": [{"class": "bottle", "confidence": 0.5, "bbox": [0, 0, "x", 1]}]})),
    ("fps not a number", lambda: ingest({**META, "fps": "fast"})),
    ("empty file", lambda: ingest(META, data=b"")),
    ("not an image", lambda: ingest(META, data=b"definitely not a jpeg")),
    ("bad extension", lambda: ingest(META, filename="frame.txt")),
]
for name, call in bad:
    r = call()
    check(f"{name} -> 400", r.status_code == 400 and r.json().get("success") is False, f"{r.status_code} {r.text[:120]}")
r = client.post("/api/edge/ingest", files={"file": ("frame.jpg", FRAME, "image/jpeg")})
check("missing meta -> rejected (422)", r.status_code == 422, str(r.status_code))
check("invalid payloads did not replace the stored frame", client.get("/api/rpi/latest").json()["hardware_metadata"]["frame_id"] == 8)

# 4. Plan A unchanged: /api/predict from the Pi client is laptop inference, no edge label
r = client.post("/api/predict", files={"file": ("f.jpg", FRAME, "image/jpeg")},
                data={"source": "raspberry_pi", "device_id": "aquascan-pi5", "camera_id": "ov5647", "frame_id": "3", "conf_threshold": "0.25"})
s = client.get("/api/rpi/latest").json()
check("Plan A /api/predict still 200 with detections", r.status_code == 200 and r.json().get("count", 0) >= 1, r.text[:160])
check("Plan A frame has no edge label (laptop inference)", "inference_location" not in (s.get("hardware_metadata") or {})
      and s["hardware_metadata"].get("source") == "raspberry_pi")

ok = all(results)
print(f"[{'PASSED' if ok else 'FAILED'}] {sum(results)}/{len(results)} edge-ingest checks")
sys.exit(0 if ok else 1)
