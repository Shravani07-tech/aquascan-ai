import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

img_path = os.path.abspath("scratch_frame.jpg")
if not os.path.exists(img_path):
    img_path = os.path.abspath("scratch/webcam_debug.jpg")

print("======================================================================")
print("   AquaScan AI — Direct FastAPI TestClient Verification")
print("======================================================================")
print(f"Target Frame File : {img_path}")

with open(img_path, "rb") as f:
    response = client.post(
        "/api/predict",
        files={"file": ("webcam_frame.jpg", f, "image/jpeg")},
        data={
            "conf_threshold": "0.20",
            "source": "laptop_webcam",
            "frame_id": "1"
        }
    )

print(f"HTTP Response Status : {response.status_code}")
json_data = response.json()
import json
print(json.dumps(json_data, indent=2))
print("======================================================================")
