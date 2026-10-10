import os
import requests
import json
from ultralytics import YOLO

def fast_scan():
    model_path = "models/best.pt"
    print(f"Loading {model_path}...")
    model = YOLO(model_path)
    
    test_dir = "dataset_clean/floating_waste_v10/test/images"
    files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)[:50]]
    
    results = model.predict(source=files, conf=0.01, verbose=False)
    
    best_file = None
    best_conf = 0.0
    
    for r in results:
        if len(r.boxes) > 0:
            top_conf = float(r.boxes[0].conf[0].cpu().numpy())
            fname = os.path.basename(r.path)
            cls_id = int(r.boxes[0].cls[0].cpu().numpy())
            print(f"File: {fname} | Top Conf: {top_conf:.4f} | Class ID: {cls_id} | Boxes: {len(r.boxes)}", flush=True)
            if top_conf > best_conf:
                best_conf = top_conf
                best_file = fname
                
    print(f"\nWinner Image: {best_file} with Confidence: {best_conf:.4f}", flush=True)
    
    if best_file:
        url = "http://localhost:8000/api/predict"
        img_path = os.path.join(test_dir, best_file)
        with open(img_path, "rb") as f:
            res = requests.post(url, files={"file": (best_file, f, "image/jpeg")}, data={"conf_threshold": round(best_conf - 0.001, 3)})
            
        print("API Response Output:")
        print(json.dumps(res.json(), indent=2), flush=True)

if __name__ == "__main__":
    fast_scan()
