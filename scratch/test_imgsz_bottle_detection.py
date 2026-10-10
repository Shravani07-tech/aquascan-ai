import os
import cv2
from ultralytics import YOLO

target_frame = os.path.abspath("scratch/webcam_debug.jpg")
if not os.path.exists(target_frame):
    target_frame = os.path.abspath("scratch_frame.jpg")

model_path = os.path.abspath("models/best.pt")
model = YOLO(model_path)
img = cv2.imread(target_frame)

print("======================================================================")
print("     AquaScan AI — Regression Test: imgsz=416 vs imgsz=640")
print("======================================================================")

for conf in [0.15, 0.20, 0.25, 0.30]:
    print(f"\n--- CONFIDENCE THRESHOLD = {conf:.2f} ---")
    
    # 640
    res_640 = model.predict(source=img, conf=conf, imgsz=640, verbose=False)
    boxes_640 = [b for r in res_640 for b in r.boxes]
    print(f"  [imgsz=640] Detections Count: {len(boxes_640)}")
    for b in boxes_640:
        c_id = int(b.cls[0].cpu().numpy())
        c_val = float(b.conf[0].cpu().numpy())
        xyxy = [round(float(v), 1) for v in b.xyxy[0].cpu().numpy()]
        print(f"     -> Class: '{model.names[c_id]}' | Conf: {c_val*100:.1f}% | BBox: {xyxy}")

    # 416
    res_416 = model.predict(source=img, conf=conf, imgsz=416, verbose=False)
    boxes_416 = [b for r in res_416 for b in r.boxes]
    print(f"  [imgsz=416] Detections Count: {len(boxes_416)}")
    for b in boxes_416:
        c_id = int(b.cls[0].cpu().numpy())
        c_val = float(b.conf[0].cpu().numpy())
        xyxy = [round(float(v), 1) for v in b.xyxy[0].cpu().numpy()]
        print(f"     -> Class: '{model.names[c_id]}' | Conf: {c_val*100:.1f}% | BBox: {xyxy}")

print("======================================================================")
