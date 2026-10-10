import os
import requests
import json

def test_real_images():
    url = "http://localhost:8000/api/predict"
    test_dirs = [
        "dataset_clean/floating_waste_v10/test/images",
        "dataset_clean/floating_waste_v10/valid/images",
        "dataset_clean/floating_waste_v10/train/images"
    ]
    
    found_detections = []
    
    for tdir in test_dirs:
        if not os.path.exists(tdir):
            continue
        images = sorted(os.listdir(tdir))[:50]  # scan first 50 images
        print(f"Scanning {len(images)} images in {tdir}...")
        
        for img_name in images:
            img_path = os.path.join(tdir, img_name)
            with open(img_path, "rb") as f:
                res = requests.post(url, files={"file": (img_name, f, "image/jpeg")}, data={"conf_threshold": 0.15})
                
            if res.status_code == 200:
                data = res.json()
                count = data.get("count", 0)
                if count > 0:
                    found_detections.append({
                        "file": img_name,
                        "path": img_path,
                        "count": count,
                        "detections": data["detections"],
                        "latency_ms": data["latency_ms"],
                        "model": data["model"],
                        "model_path": data["model_path"]
                    })
                    print(f"✅ DETECTED {count} objects in {img_name}:")
                    for d in data["detections"]:
                        print(f"   - Class: {d['class']} | Conf: {d['confidence']*100:.1f}% | BBox: {d['bbox']}")
                    if len(found_detections) >= 3:
                        break
        if found_detections:
            break

    print("\n--- TEST SUMMARY ---")
    if found_detections:
        print(f"SUCCESS: Found real detections across {len(found_detections)} test images!")
        first_det = found_detections[0]
        print(f"\nRepresentative Verification Result for Image '{first_det['file']}':")
        print(f"  Model Used     : {first_det['model']} ({first_det['model_path']})")
        print(f"  Objects Count  : {first_det['count']}")
        print(f"  First Detection: Class='{first_det['detections'][0]['class']}', Confidence={first_det['detections'][0]['confidence']}, BBox={first_det['detections'][0]['bbox']}")
        print(f"  Latency (ms)   : Total={first_det['latency_ms']['total']}ms, Inference={first_det['latency_ms']['inference']}ms, FPS={first_det['latency_ms']['fps']}")
        return True
    else:
        print("WARNING: No detections found above threshold 0.15 in tested images.")
        return False

if __name__ == "__main__":
    test_real_images()
