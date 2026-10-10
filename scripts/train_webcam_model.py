import os
import sys
import glob
from ultralytics import YOLO

BASE_DIR = os.path.abspath("dataset_webcam")
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
BASE_MODEL = os.path.abspath("models/best.pt")
OUTPUT_MODEL = os.path.abspath("models/best_webcam.pt")

def train_webcam_adaptation():
    print("======================================================================")
    print("           AquaScan AI — Webcam Model Fine-Tuning Pipeline")
    print("======================================================================")
    
    if not os.path.exists(BASE_MODEL):
        print(f"[ERROR] Base model not found at {BASE_MODEL}")
        sys.exit(1)

    train_imgs = glob.glob(os.path.join(BASE_DIR, "train", "images", "*.jpg")) + glob.glob(os.path.join(BASE_DIR, "train", "images", "*.png"))
    if len(train_imgs) == 0:
        print("[AquaScan Training Notice]")
        print(f"  - Base model preserved: {BASE_MODEL}")
        print(f"  - Dataset structure ready at: {BASE_DIR}")
        print("  - Train images count: 0")
        print("  - Action required: Place annotated webcam training images in dataset_webcam/train/ and dataset_webcam/val/ before executing fine-tuning.")
        print("======================================================================")
        return

    print(f"Base Model    : {BASE_MODEL}")
    print(f"Target Dataset : {DATA_YAML}")
    print(f"Output Target  : {OUTPUT_MODEL}")
    print("Starting controlled CPU fine-tuning (5 epochs)...")

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=5,
        imgsz=640,
        batch=4,
        workers=0,
        device="cpu",
        name="webcam_finetune",
        exist_ok=True,
        verbose=True
    )

    # Save fine-tuned weights to models/best_webcam.pt
    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"[SUCCESS] Fine-tuned model saved to: {OUTPUT_MODEL}")
    else:
        print(f"[WARNING] Fine-tuning finished but best weights not found at {best_weights}")

if __name__ == "__main__":
    train_webcam_adaptation()
