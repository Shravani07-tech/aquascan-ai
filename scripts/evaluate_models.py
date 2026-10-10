import os
import sys
import glob
import cv2
from ultralytics import YOLO

MODEL_A_PATH = os.path.abspath("models/best.pt")
MODEL_B_PATH = os.path.abspath("models/best_webcam.pt")

MARINE_TEST_DIR = os.path.abspath("dataset_clean/floating_waste_v10/test/images")
WEBCAM_VAL_DIR = os.path.abspath("dataset_webcam/val/images")
# Fixed committed test frame, so results never depend on the rolling scratch/webcam_debug.jpg
TEST_FRAME_PATH = os.path.abspath("scratch_frame.jpg")

CLASSES = ['bottle', 'can', 'carton', 'paper', 'plastic']

def evaluate_model(model_path, name):
    if not os.path.exists(model_path):
        print(f"[{name}] Model file not found: {model_path}")
        return None

    print(f"\n======================================================================")
    print(f"       EVALUATING MODEL: {name} ({os.path.basename(model_path)})")
    print(f"======================================================================")
    model = YOLO(model_path)
    print(f"Model classes: {model.names}")

    # 1. Marine Test Dataset Evaluation
    marine_imgs = sorted(glob.glob(os.path.join(MARINE_TEST_DIR, "*.jpg")) + glob.glob(os.path.join(MARINE_TEST_DIR, "*.png")))[:20]
    marine_detections = 0
    marine_total_conf = 0.0

    for img_p in marine_imgs:
        img = cv2.imread(img_p)
        res = model.predict(source=img, conf=0.25, verbose=False)
        boxes = [b for r in res for b in r.boxes]
        marine_detections += len(boxes)
        if boxes:
            marine_total_conf += sum(float(b.conf[0].cpu().numpy()) for b in boxes) / len(boxes)

    print(f"\n1. Marine Test Set Evaluation (Sample n={len(marine_imgs)}):")
    print(f"   - Total Detections Count : {marine_detections}")
    print(f"   - Avg Detection Conf     : {marine_total_conf/max(1, len(marine_imgs))*100:.1f}%")

    # 2. Debug Webcam Frame Evaluation
    target_webcam = TEST_FRAME_PATH
    if os.path.exists(target_webcam):
        img = cv2.imread(target_webcam)
        res = model.predict(source=img, conf=0.25, verbose=False)
        boxes = [b for r in res for b in r.boxes]
        print(f"\n2. Webcam Frame Evaluation ({os.path.basename(target_webcam)}):")
        print(f"   - Detections Count at conf=0.25: {len(boxes)}")
        for i, b in enumerate(boxes, 1):
            c_id = int(b.cls[0].cpu().numpy())
            c_val = float(b.conf[0].cpu().numpy())
            xyxy = [round(float(v), 1) for v in b.xyxy[0].cpu().numpy()]
            print(f"     {i}. [{CLASSES[c_id].upper()}] Conf: {c_val*100:.1f}% | BBox: {xyxy}")

    return {
        "name": name,
        "marine_detections": marine_detections
    }

def main():
    print("======================================================================")
    print("            AquaScan AI — Dual Model Comparison Suite")
    print("======================================================================")
    
    res_a = evaluate_model(MODEL_A_PATH, "Original Marine Model (best.pt)")
    res_b = evaluate_model(MODEL_B_PATH, "Webcam Adapted Model (best_webcam.pt)")

    if res_a and res_b:
        print("\n======================================================================")
        print("                   COMPARISON & DEGRADATION CHECK")
        print("======================================================================")
        print(f"Original best.pt Marine Detections       : {res_a['marine_detections']}")
        print(f"Webcam best_webcam.pt Marine Detections  : {res_b['marine_detections']}")
        deg = (res_a['marine_detections'] - res_b['marine_detections']) / max(1, res_a['marine_detections']) * 100
        print(f"Marine Performance Change                : {-deg:.1f}%")
        if deg > 15.0:
            print("[WARNING] Catastrophic degradation detected on marine test set!")
        else:
            print("[PASSED] Marine detection performance preserved within tolerance.")
        print("======================================================================")

if __name__ == "__main__":
    main()
