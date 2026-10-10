import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import CLASS_NAMES, MAX_FILE_SIZE

def test_complete_api_suite():
    client = TestClient(app)
    print("=" * 65)
    print("      AquaScan AI — Comprehensive API Automated Test Suite")
    print("=" * 65)

    # 1. GET /api/status
    print("\n--- 1. Testing GET /api/status ---")
    res = client.get("/api/status")
    assert res.status_code == 200, f"Status API failed: {res.status_code}"
    status_data = res.json()
    print("  [+] Status API Output:", status_data)
    assert status_data["success"] == True
    assert status_data["status"] == "ready"
    assert "model_path" in status_data
    assert "device" in status_data
    assert "cuda_available" in status_data
    assert status_data["classes_count"] == 5
    assert status_data["classes"] == CLASS_NAMES
    assert status_data["is_fallback"] == False

    # 2. GET /api/samples
    print("\n--- 2. Testing GET /api/samples ---")
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples_data = res.json()
    samples = samples_data.get("samples", [])
    print(f"  [+] Retrieved {len(samples)} samples")
    assert len(samples) > 0
    sample_id = samples[0]["id"]
    assert "image_url" in samples[0]
    print(f"  [+] Sample 1 Image URL: {samples[0]['image_url']}")

    # 3. GET /api/samples/{sample_id}/image
    print(f"\n--- 3. Testing GET /api/samples/{sample_id}/image Thumbnail ---")
    res = client.get(f"/api/samples/{sample_id}/image")
    assert res.status_code == 200, f"Thumbnail fetch failed: {res.status_code}"
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 0
    print(f"  [+] Thumbnail Image Loaded: {len(res.content)} bytes")

    # 4. POST /api/predict with sample_id
    print(f"\n--- 4. Testing POST /api/predict with sample_id='{sample_id}' ---")
    res = client.post("/api/predict", data={"sample_id": sample_id, "conf_threshold": 0.01})
    assert res.status_code == 200
    pred_data = res.json()
    assert pred_data["success"] == True
    assert "detections" in pred_data
    assert "latency_ms" in pred_data
    assert "image" in pred_data
    assert "category_counts" in pred_data
    assert pred_data["is_fallback"] == False
    print(f"  [+] Sample Prediction Detections: {pred_data['count']} | Latency: {pred_data['latency_ms']['total']} ms")

    # 5. POST /api/predict with file upload
    print("\n--- 5. Testing POST /api/predict with Valid File Upload ---")
    test_img = os.path.abspath("dataset_clean/floating_waste_v10/test/images/345_jpg.rf.30c35b52c72364d004c61b985838d035.jpg")
    if not os.path.exists(test_img):
        test_img = "test.jpg"

    assert os.path.exists(test_img), f"Test image not found at {test_img}"
    with open(test_img, "rb") as f:
        res = client.post("/api/predict", files={"file": ("test.jpg", f, "image/jpeg")}, data={"conf_threshold": 0.01})
    
    assert res.status_code == 200
    file_pred = res.json()
    assert file_pred["success"] == True
    assert file_pred["is_fallback"] == False
    print(f"  [+] File Upload Prediction Detections: {file_pred['count']} | Latency: {file_pred['latency_ms']['total']} ms")

    # 6. Real Known-Positive Detection Verification
    print("\n--- 6. Testing Real Known-Positive Detection Assertions ---")
    assert file_pred["count"] > 0, "Expected at least 1 detection on known-positive image!"
    det = file_pred["detections"][0]
    assert det["class"] in CLASS_NAMES
    assert det["confidence"] > 0
    assert len(det["bbox"]) == 4
    assert len(det["bbox_normalized"]) == 4
    assert 0.0 <= det["bbox_normalized"][0] <= 1.0
    assert 0.0 <= det["bbox_normalized"][1] <= 1.0
    print(f"  [+] Detection Verified: Class='{det['class']}', Conf={det['confidence']}, BBox={det['bbox']}")

    # 7. Hardware Metadata Verification
    print("\n--- 7. Testing Optional Hardware Metadata Parameters ---")
    with open(test_img, "rb") as f:
        res = client.post(
            "/api/predict",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={
                "conf_threshold": 0.01,
                "source": "raspberry_pi",
                "device_id": "aquascan-pi-01",
                "camera_id": "picam-v2",
                "frame_id": 42
            }
        )
    assert res.status_code == 200
    hw_pred = res.json()
    assert "hardware_metadata" in hw_pred
    assert hw_pred["hardware_metadata"]["source"] == "raspberry_pi"
    assert hw_pred["hardware_metadata"]["device_id"] == "aquascan-pi-01"
    assert hw_pred["hardware_metadata"]["frame_id"] == 42
    print("  [+] Hardware Metadata Echo Verified:", hw_pred["hardware_metadata"])

    # 8. Invalid Extension Handling
    print("\n--- 8. Testing Invalid File Extension Handling (.txt) ---")
    res = client.post("/api/predict", files={"file": ("test.txt", b"invalid content", "text/plain")})
    assert res.status_code == 400
    err_json = res.json()
    assert err_json["success"] == False
    assert err_json["error_type"] == "invalid_extension"
    print("  [+] Extension Error Response:", err_json)

    # 9. Oversized Upload Handling (>15MB)
    print("\n--- 9. Testing Oversized File Upload Handling (>15MB) ---")
    huge_bytes = b"0" * (16 * 1024 * 1024)  # 16 MB
    res = client.post("/api/predict", files={"file": ("huge.png", huge_bytes, "image/png")})
    assert res.status_code == 400
    err_json = res.json()
    assert err_json["success"] == False
    assert err_json["error_type"] == "oversized_upload"
    print("  [+] Oversized Error Response:", err_json)

    # 10. Empty File Handling (0 bytes)
    print("\n--- 10. Testing Empty File Upload Handling (0 bytes) ---")
    res = client.post("/api/predict", files={"file": ("empty.png", b"", "image/png")})
    assert res.status_code == 400
    err_json = res.json()
    assert err_json["success"] == False
    assert err_json["error_type"] == "empty_file"
    print("  [+] Empty File Error Response:", err_json)

    # 11. Invalid Sample ID Handling
    print("\n--- 11. Testing Invalid Sample ID Handling ---")
    res = client.post("/api/predict", data={"sample_id": "nonexistent_sample_999"})
    assert res.status_code == 404
    err_json = res.json()
    assert err_json["success"] == False
    assert err_json["error_type"] == "sample_not_found"
    print("  [+] Sample Not Found Error Response:", err_json)

    # 12. Invalid Confidence Threshold Handling (<0.0 or >1.0)
    print("\n--- 12. Testing Invalid Confidence Threshold Handling ---")
    res = client.post("/api/predict", data={"sample_id": sample_id, "conf_threshold": 1.5})
    assert res.status_code == 400
    err_json = res.json()
    assert err_json["success"] == False
    assert err_json["error_type"] == "invalid_confidence"
    print("  [+] Invalid Confidence Error Response:", err_json)

    # 13. Missing Input Handling
    print("\n--- 13. Testing Missing Input Handling ---")
    res = client.post("/api/predict")
    assert res.status_code == 400
    err_json = res.json()
    assert err_json["success"] == False
    assert err_json["error_type"] == "missing_input"
    print("  [+] Missing Input Error Response:", err_json)

    print("\n" + "=" * 65)
    print("   [PASSED] ALL 13 API TEST SUITE ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True

if __name__ == "__main__":
    test_complete_api_suite()
