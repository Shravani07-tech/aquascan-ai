import os
import sys

sys.path.insert(0, os.path.abspath("."))

import time
from fastapi.testclient import TestClient
from backend.main import app
from hardware.rpi_client import CameraBackend


print("======================================================================")
print("   AquaScan AI — Raspberry Pi Edge Streaming Phase 2 Verification")
print("======================================================================")

client = TestClient(app)

# 1. Verify Status Handshake
res_status = client.get("/api/status")
assert res_status.status_code == 200
status_data = res_status.json()
print("Handshake Response:")
print(f"  - Model       : {status_data['model']} (Fallback: {status_data['is_fallback']})")
print(f"  - Model Path  : {status_data['model_path']}")
print(f"  - Classes ({status_data['classes_count']}): {status_data['classes']}")

# 2. Camera Backend Initialization
cam = CameraBackend(mode="auto", width=640, height=480, fallback_image="test.jpg")
print(f"\nInitialized Camera Backend: {cam.active_backend.upper()}")

# 3. Simulate 10 Consecutive Frame Transmission Loop
print("\nTransmitting 10 consecutive edge frames...")
total_objects = 0

for frame_id in range(1, 11):
    jpeg_bytes = cam.capture_jpeg_bytes()
    
    data_payload = {
        "conf_threshold": "0.25",
        "source": "raspberry_pi",
        "device_id": "aquascan-pi-demo",
        "camera_id": "picam-v2",
        "frame_id": str(frame_id),
        "timestamp": f"{time.time():.4f}"
    }

    files_payload = {
        "file": (f"frame_{frame_id:06d}.jpg", jpeg_bytes, "image/jpeg")
    }

    t0 = time.time()
    res = client.post("/api/predict", data=data_payload, files=files_payload)
    rtt_ms = (time.time() - t0) * 1000.0

    assert res.status_code == 200, f"Frame {frame_id} failed with status {res.status_code}"
    result = res.json()

    assert result["success"] == True
    assert result["is_fallback"] == False
    assert result["hardware_metadata"]["frame_id"] == frame_id
    assert result["hardware_metadata"]["source"] == "raspberry_pi"

    cnt = result["count"]
    total_objects += cnt
    lat = result["latency_ms"]["total"]

    print(f"  [FRAME #{frame_id:02d}] HTTP 200 OK | Objects: {cnt:<2d} | Inference: {lat}ms | RTT: {rtt_ms:.1f}ms | Frame ID Echo: {result['hardware_metadata']['frame_id']}")

cam.release()

print("\n======================================================================")
print(f"  Total Frames Sent : 10 / 10")
print(f"  Total Detections  : {total_objects}")
print("  [PASSED] PHASE 2 EDGE STREAMING VERIFICATION PASSED 100%!")
print("======================================================================")
