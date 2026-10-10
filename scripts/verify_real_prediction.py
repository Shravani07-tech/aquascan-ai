import os
import requests
import json

def verify_prediction():
    url = "http://localhost:8000/api/predict"
    img_path = "dataset_clean/floating_waste_v10/test/images/108_jpg.rf.f4dabaf833855f434ada356b1f937b53.jpg"
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return
        
    print(f"Testing real test image '{os.path.basename(img_path)}' via POST /api/predict...")
    with open(img_path, "rb") as f:
        res = requests.post(url, files={"file": (os.path.basename(img_path), f, "image/jpeg")}, data={"conf_threshold": 0.20})
        
    print(f"HTTP Status Code: {res.status_code}")
    data = res.json()
    print("API Response Output:")
    print(json.dumps(data, indent=2))
    
    assert res.status_code == 200
    assert data["success"] == True
    assert data["count"] > 0
    assert "detections" in data
    assert "latency_ms" in data
    assert "image" in data
    print("\n[SUCCESS] VERIFICATION PASSED: Real marine waste image produced verified detections!")

if __name__ == "__main__":
    verify_prediction()
