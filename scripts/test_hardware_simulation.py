import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app


def simulate_raspberry_pi_edge_transmission():
    print("=" * 70)
    print("   AquaScan AI — Raspberry Pi Edge Hardware Simulation Test")
    print("=" * 70)

    client = TestClient(app)

    # 1. Verify Status API reachability
    res_status = client.get("/api/status")
    assert res_status.status_code == 200, f"Status endpoint failed: {res_status.status_code}"
    status_data = res_status.json()
    print("  [+] Edge Status Handshake:")
    print(f"      - Active Model : {status_data.get('model')}")
    print(f"      - Model Path   : {status_data.get('model_path')}")
    print(f"      - Is Fallback  : {status_data.get('is_fallback')}")
    print(f"      - Device       : {status_data.get('device')}")

    assert status_data["success"] == True
    assert status_data["is_fallback"] == False

    # 2. Select actual test image frame
    test_img = os.path.abspath("dataset_clean/floating_waste_v10/test/images/345_jpg.rf.30c35b52c72364d004c61b985838d035.jpg")
    if not os.path.exists(test_img):
        test_img = "test.jpg"

    assert os.path.exists(test_img), f"Simulation test image not found at {test_img}"

    # 3. Formulate Raspberry Pi frame transmission payload with hardware metadata
    simulated_metadata = {
        "source": "raspberry_pi",
        "device_id": "aquascan-pi-demo",
        "camera_id": "picamera-demo",
        "frame_id": 1,
        "timestamp": str(time.time()),
        "conf_threshold": 0.01
    }

    print(f"\n  [*] Simulating Raspberry Pi Camera Frame Transmission:")
    print(f"      - Image Frame : {os.path.basename(test_img)} ({os.path.getsize(test_img)} bytes)")
    print(f"      - Source      : {simulated_metadata['source']}")
    print(f"      - Device ID   : {simulated_metadata['device_id']}")
    print(f"      - Camera ID   : {simulated_metadata['camera_id']}")
    print(f"      - Frame ID    : {simulated_metadata['frame_id']}")
    print(f"      - Timestamp   : {simulated_metadata['timestamp']}")

    t0 = time.time()
    with open(test_img, "rb") as f:
        res = client.post(
            "/api/predict",
            files={"file": (os.path.basename(test_img), f, "image/jpeg")},
            data=simulated_metadata
        )
    rtt_ms = (time.time() - t0) * 1000.0

    assert res.status_code == 200, f"Predict API failed: {res.status_code} - {res.text}"
    data = res.json()

    print("\n" + "=" * 70)
    print("           SIMULATED RASPBERRY PI DETECTION RESPONSE")
    print("=" * 70)
    print(f"  Response Status   : {res.status_code} OK")
    print(f"  Success Flag      : {data.get('success')}")
    print(f"  Model             : {data.get('model')} (Fallback: {data.get('is_fallback')})")
    print(f"  Objects Count     : {data.get('count')}")
    print(f"  Category Counts   : {data.get('category_counts')}")

    detections = data.get("detections", [])
    if detections:
        print(f"\n  Detected Objects:")
        for idx, d in enumerate(detections, 1):
            print(f"    {idx}. Class: '{d['class']}' | Confidence: {d['confidence']*100:.1f}% | BBox (px): {d['bbox']}")

    latency = data.get("latency_ms", {})
    print(f"\n  Performance Breakdown:")
    print(f"    - Preprocessing : {latency.get('preprocess')} ms")
    print(f"    - Inference     : {latency.get('inference')} ms")
    print(f"    - Postprocessing: {latency.get('postprocess')} ms")
    print(f"    - Total Server  : {latency.get('total')} ms ({latency.get('fps')} FPS)")
    print(f"    - Total RTT     : {round(rtt_ms, 2)} ms")

    hw_echo = data.get("hardware_metadata", {})
    print(f"\n  Validated Hardware Metadata Echo:")
    print(f"    - Source   : {hw_echo.get('source')}")
    print(f"    - Device ID: {hw_echo.get('device_id')}")
    print(f"    - Camera ID: {hw_echo.get('camera_id')}")
    print(f"    - Frame ID : {hw_echo.get('frame_id')}")

    # Validations
    assert data["success"] == True
    assert data["is_fallback"] == False
    assert hw_echo["source"] == "raspberry_pi"
    assert hw_echo["device_id"] == "aquascan-pi-demo"
    assert hw_echo["camera_id"] == "picamera-demo"
    assert hw_echo["frame_id"] == 1

    print("=" * 70)
    print("   [PASSED] SOFTWARE-ONLY HARDWARE INTEGRATION SIMULATION SUCCESSFUL!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    simulate_raspberry_pi_edge_transmission()
