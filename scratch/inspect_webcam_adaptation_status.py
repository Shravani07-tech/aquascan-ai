import os
import sys
import glob
import time
import cv2
from ultralytics import YOLO

dataset_dir = os.path.abspath("dataset_webcam")
train_img_dir = os.path.join(dataset_dir, "train", "images")
train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
val_img_dir = os.path.join(dataset_dir, "val", "images")
val_lbl_dir = os.path.join(dataset_dir, "val", "labels")

CLASSES = ['bottle', 'can', 'carton', 'paper', 'plastic']

print("======================================================================")
print("     AquaScan AI — Webcam Dataset & Model Inspection Report")
print("======================================================================")

# 1. Dataset Inspection
train_imgs = glob.glob(os.path.join(train_img_dir, "*.jpg")) + glob.glob(os.path.join(train_img_dir, "*.png"))
val_imgs = glob.glob(os.path.join(val_img_dir, "*.jpg")) + glob.glob(os.path.join(val_img_dir, "*.png"))

train_lbls = glob.glob(os.path.join(train_lbl_dir, "*.txt"))
val_lbls = glob.glob(os.path.join(val_lbl_dir, "*.txt"))

print(f"1. Dataset Image Counts:")
print(f"   - train/images: {len(train_imgs)}")
print(f"   - val/images  : {len(val_imgs)}")

class_counts = {c: 0 for c in CLASSES}
class_counts['background'] = 0
total_objects = 0

for lbl_p in train_lbls + val_lbls:
    with open(lbl_p, "r") as f:
        lines = [l.strip().split() for l in f if l.strip()]
    if not lines:
        class_counts['background'] += 1
    else:
        for l in lines:
            cid = int(l[0])
            total_objects += 1
            if 0 <= cid < len(CLASSES):
                class_counts[CLASSES[cid]] += 1

print(f"\n2. Labelled Object Counts:")
print(f"   - Total Labelled Objects: {total_objects}")
for c in CLASSES:
    print(f"   - {c:10s}: {class_counts[c]} instances")
print(f"   - background: {class_counts['background']} empty images")

# 2. Model File Inspection
model_a = os.path.abspath("models/best.pt")
model_b = os.path.abspath("models/best_webcam.pt")

print(f"\n3. Model Files Inspection:")
if os.path.exists(model_a):
    st_a = os.stat(model_a)
    mtime_a = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st_a.st_mtime))
    print(f"   - Original Model (best.pt): Exists | Size: {st_a.st_size} bytes | Modified: {mtime_a}")
else:
    print(f"   - Original Model (best.pt): NOT FOUND")

if os.path.exists(model_b):
    st_b = os.stat(model_b)
    mtime_b = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st_b.st_mtime))
    print(f"   - Webcam Model (best_webcam.pt): Exists | Size: {st_b.st_size} bytes | Modified: {mtime_b}")
else:
    print(f"   - Webcam Model (best_webcam.pt): DOES NOT EXIST YET (Pending training images)")

# 3. Model Detections Evaluation on Webcam Frame
# Fixed committed test frame, so results never depend on the rolling scratch/webcam_debug.jpg
target_frame = os.path.abspath("scratch_frame.jpg")

print(f"\n4. Model Evaluations on Saved Webcam Frame ({target_frame}):")
thresholds = [0.01, 0.05, 0.10, 0.15, 0.25]

def eval_frame(m_path, m_name):
    if not os.path.exists(m_path):
        print(f"\n   [{m_name}] Model file not found at {m_path}")
        return
    print(f"\n   --- Evaluation for {m_name} ({os.path.basename(m_path)}) ---")
    img = cv2.imread(target_frame)
    model = YOLO(m_path)
    for conf in thresholds:
        res = model.predict(source=img, conf=conf, verbose=False)
        boxes = [b for r in res for b in r.boxes]
        print(f"   [Conf={conf:.2f}] Detections Count: {len(boxes)}")
        for i, b in enumerate(boxes, 1):
            c_id = int(b.cls[0].cpu().numpy())
            c_val = float(b.conf[0].cpu().numpy())
            xyxy = [round(float(v), 1) for v in b.xyxy[0].cpu().numpy()]
            print(f"      {i}. Class: '{CLASSES[c_id]}' | Conf: {c_val*100:.2f}% | BBox: {xyxy}")

eval_frame(model_a, "Original best.pt")
eval_frame(model_b, "Adapted best_webcam.pt")

print("\n======================================================================")
