import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

target_frame = os.path.abspath("scratch/webcam_debug.jpg")
if not os.path.exists(target_frame):
    target_frame = os.path.abspath("scratch_frame.jpg")

model_path = os.path.abspath("models/best.pt")
model = YOLO(model_path)

print("======================================================================")
print("     AquaScan AI — Regression Investigation: Pipeline Inputs")
print("======================================================================")
print(f"Target Frame: {target_frame}")
print(f"Model Path  : {model_path}")

# Load in different ways
cv2_bgr = cv2.imread(target_frame)
cv2_rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.open(target_frame)

pipelines = {
    "1. Direct File Path String": target_frame,
    "2. OpenCV BGR NumPy Array": cv2_bgr,
    "3. OpenCV RGB NumPy Array": cv2_rgb,
    "4. PIL Image Object": pil_img
}

for conf in [0.01, 0.05, 0.10, 0.15, 0.25]:
    print(f"\n==================== CONFIDENCE THRESHOLD = {conf:.2f} ====================")
    for name, source_input in pipelines.items():
        res = model.predict(source=source_input, conf=conf, verbose=False)
        boxes = [b for r in res for b in r.boxes]
        print(f"\n[{name}] -> Detections Count: {len(boxes)}")
        for i, b in enumerate(boxes, 1):
            c_id = int(b.cls[0].cpu().numpy())
            c_val = float(b.conf[0].cpu().numpy())
            xyxy = [round(float(v), 1) for v in b.xyxy[0].cpu().numpy()]
            print(f"   {i}. Class: '{model.names[c_id]}' (ID: {c_id}) | Conf: {c_val*100:.2f}% | BBox: {xyxy}")

print("\n======================================================================")
print("           TESTING IMGSZ PARAMETER VARIATIONS (conf=0.25)")
print("======================================================================")
for imgsz_val in [320, 416, 640, 1280]:
    res = model.predict(source=cv2_bgr, conf=0.25, imgsz=imgsz_val, verbose=False)
    boxes = [b for r in res for b in r.boxes]
    print(f"\n[imgsz={imgsz_val}] -> Detections Count: {len(boxes)}")
    for i, b in enumerate(boxes, 1):
        c_id = int(b.cls[0].cpu().numpy())
        c_val = float(b.conf[0].cpu().numpy())
        xyxy = [round(float(v), 1) for v in b.xyxy[0].cpu().numpy()]
        print(f"   {i}. Class: '{model.names[c_id]}' | Conf: {c_val*100:.2f}% | BBox: {xyxy}")
