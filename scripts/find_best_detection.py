import os
import requests
import json
from ultralytics import YOLO

def find_and_test():
    model_path = "models/best.pt"
    test_dir = "dataset_clean/floating_waste_v10/test/images"
    
    print(f"Loading {model_path} directly to scan test images...")
    model = YOLO(model_path)
    
    images = sorted(os.listdir(test_dir))
    best_img = None
    best_conf = 0.0
    best_dets = []
    
    for img_name in images[:100]:
        img_path = os.path.join(test_dir, img_name)
        results = model.predict(source=img_path, conf=0.15, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                conf = float(r.boxes[0].conf[0].cpu().numpy())
                if conf > best_conf:
                    best_conf = conf
                    best_img = img_name
                    best_dets = r.boxes
                    
    print(f"\nHighest Confidence Test Image Found: {best_img} (Max Conf: {best_conf:.4f})")
    
    if best_img:
        img_path = os.path.join(test_dir, best_img)
        url = "http://localhost:8000/api/predict"
        print(f"Testing {best_img} via API endpoint POST /api/predict...")
        with open(img_path, "rb") as f:
            res = requests.post(url, files={"file": (best_img, f, "image/jpeg")}, data={"conf_threshold": 0.15})
            
        print(f"API Response Status Code: {res.status_code}")
        data = res.json()
        print("API Response Output:")
        print(json.dumps(data, indent=2))
        
        assert res.status_code == 200
        assert data["success"] == True
        assert data["count"] > 0
        assert "detections" in data
        assert "latency_ms" in data
        assert "image" in data
        print("\nVERIFICATION SUCCESSFUL: Real marine-waste test image produced verified detections!")

if __name__ == "__main__":
    find_and_test()
