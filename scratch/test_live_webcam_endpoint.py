import os
import requests

url = "http://localhost:8000/api/predict"
img_path = os.path.abspath("scratch_frame.jpg")

if not os.path.exists(img_path):
    img_path = os.path.abspath("scratch/webcam_debug.jpg")

print("======================================================================")
print(f"  AquaScan AI — Live Webcam API Endpoint Direct Verification")
print("======================================================================")
print(f"Target Image File : {img_path}")
print(f"Target Endpoint   : {url}")

with open(img_path, "rb") as f:
    files = {"file": ("webcam_frame.jpg", f, "image/jpeg")}
    data = {
        "conf_threshold": "0.20",
        "source": "laptop_webcam",
        "frame_id": "1"
    }
    try:
        res = requests.post(url, files=files, data=data, timeout=5)
        print(f"HTTP Response Status : {res.status_code}")
        json_data = res.json()
        print("\nComplete JSON Response:")
        import json
        print(json.dumps(json_data, indent=2))
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
print("======================================================================")
