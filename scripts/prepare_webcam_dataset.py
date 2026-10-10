import os
import sys
import glob
import cv2
import random

BASE_DIR = os.path.abspath("dataset_webcam")
TRAIN_IMG = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL = os.path.join(BASE_DIR, "train", "labels")
VAL_IMG = os.path.join(BASE_DIR, "val", "images")
VAL_LBL = os.path.join(BASE_DIR, "val", "labels")

CLASSES = ['bottle', 'can', 'carton', 'paper', 'plastic']

def init_dataset_structure():
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        os.makedirs(d, exist_ok=True)
    print(f"[AquaScan Dataset] Initialized dataset directories under: {BASE_DIR}")

def check_dataset_status():
    init_dataset_structure()
    train_imgs = glob.glob(os.path.join(TRAIN_IMG, "*.jpg")) + glob.glob(os.path.join(TRAIN_IMG, "*.png"))
    val_imgs = glob.glob(os.path.join(VAL_IMG, "*.jpg")) + glob.glob(os.path.join(VAL_IMG, "*.png"))
    
    print("\n======================================================================")
    print("           AquaScan AI — Webcam Dataset Status Summary")
    print("======================================================================")
    print(f"Train Images Count: {len(train_imgs)}")
    print(f"Val Images Count  : {len(val_imgs)}")
    
    class_counts = {c: 0 for c in CLASSES}
    class_counts['background'] = 0
    
    all_lbls = glob.glob(os.path.join(TRAIN_LBL, "*.txt")) + glob.glob(os.path.join(VAL_LBL, "*.txt"))
    for lbl in all_lbls:
        with open(lbl, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]
        if not lines:
            class_counts['background'] += 1
        else:
            for l in lines:
                cid = int(l[0])
                if 0 <= cid < len(CLASSES):
                    class_counts[CLASSES[cid]] += 1

    print("\nAnnotation Counts by Class:")
    for c in CLASSES:
        print(f"  - {c:10s}: {class_counts[c]} instances")
    print(f"  - background: {class_counts['background']} images")
    print("======================================================================")

if __name__ == "__main__":
    check_dataset_status()
