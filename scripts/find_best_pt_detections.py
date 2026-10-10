import os
from ultralytics import YOLO

def scan_model():
    model_path = "models/best.pt"
    print(f"Scanning with model: {model_path}")
    model = YOLO(model_path)
    
    test_dirs = [
        "dataset_clean/floating_waste_v10/test/images",
        "dataset_clean/floating_waste_v10/valid/images"
    ]
    
    matches = []
    for tdir in test_dirs:
        if not os.path.exists(tdir):
            continue
        print(f"Scanning directory: {tdir}...")
        for fname in os.listdir(tdir):
            fpath = os.path.join(tdir, fname)
            results = model.predict(source=fpath, conf=0.05, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    top_conf = float(r.boxes[0].conf[0].cpu().numpy())
                    cls_id = int(r.boxes[0].cls[0].cpu().numpy())
                    matches.append((top_conf, fname, fpath, cls_id, len(r.boxes)))
                    
    matches.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTotal images with detections: {len(matches)}")
    print("Top 10 Detections:")
    for conf, fname, fpath, cls_id, num_boxes in matches[:10]:
        print(f"  - Conf: {conf:.4f} | Class ID: {cls_id} | Boxes: {num_boxes} | File: {fname}")
        
    if matches:
        return matches[0]
    return None

if __name__ == "__main__":
    scan_model()
