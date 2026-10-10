import os
import time
import shutil
import cv2
import csv
import torch
from ultralytics import YOLO

def main():
    # Detect device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Target Execution Device: {device} (CUDA Available: {torch.cuda.is_available()})")
    
    clean_dir = os.path.abspath("dataset_clean/floating_waste_v10")
    data_yaml = os.path.join(clean_dir, "data.yaml")
    
    # Ensure data.yaml has the current absolute path
    yaml_content = f"""path: {clean_dir.replace('\\', '/')}
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['bottle', 'can', 'carton', 'paper', 'plastic']

roboflow:
  workspace: garbagedetection-gqqya
  project: floating-waste
  version: 10
  license: CC BY 4.0
  url: https://universe.roboflow.com/garbagedetection-gqqya/floating-waste/dataset/10
"""
    with open(data_yaml, "w") as f:
        f.write(yaml_content)

    project_dir = os.path.abspath("outputs/training")
    run_name = "yolov8n_full_30ep"
    run_dir = os.path.join(project_dir, run_name)
    
    if os.path.exists(run_dir):
        print(f"Cleaning previous run output directory: {run_dir}")
        shutil.rmtree(run_dir)
        
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    print(f"Starting FULL Training on {device} (30 epochs, 6,687 training images, imgsz=640)...")
    start_train_time = time.time()
    
    # Train on complete dataset (fraction=1.0)
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16 if torch.cuda.is_available() else 4,
        workers=4 if torch.cuda.is_available() else 2,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        fraction=1.0,
        save=True,
        plots=True
    )
    
    end_train_time = time.time()
    total_training_time_sec = end_train_time - start_train_time
    print(f"Training completed in {total_training_time_sec:.2f} seconds ({total_training_time_sec/60:.2f} minutes).")
    
    # Load best weights
    best_weights_path = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_weights_path):
        best_weights_path = os.path.join(run_dir, "weights", "last.pt")
        
    print(f"Loading best weights from {best_weights_path}...")
    best_model = YOLO(best_weights_path)
    
    # Validation Evaluation
    print("Evaluating on Validation Set (638 images)...")
    val_results = best_model.val(data=data_yaml, split="val")
    
    # Test Evaluation
    print("Evaluating on Test Set (319 images)...")
    test_results = best_model.val(data=data_yaml, split="test")
    
    # Predict on test images & benchmark speed
    predictions_dir = os.path.join(run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    test_img_dir = os.path.join(clean_dir, "test", "images")
    test_images = sorted(os.listdir(test_img_dir))[:10] # 10 test images
    
    speed_pre, speed_inf, speed_post = [], [], []
    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        res = best_model.predict(source=img_path, device=device, verbose=False)
        for r in res:
            speed_pre.append(r.speed.get("preprocess", 0.0))
            speed_inf.append(r.speed.get("inference", 0.0))
            speed_post.append(r.speed.get("postprocess", 0.0))
            annotated_img = r.plot()
            cv2.imwrite(os.path.join(predictions_dir, img_name), annotated_img)
            
    avg_pre = sum(speed_pre) / len(speed_pre) if speed_pre else 0.0
    avg_inf = sum(speed_inf) / len(speed_inf) if speed_inf else 0.0
    avg_post = sum(speed_post) / len(speed_post) if speed_post else 0.0
    avg_tot = avg_pre + avg_inf + avg_post
    avg_fps = 1000.0 / avg_tot if avg_tot > 0 else 0.0
    
    print("\nFull 30-epoch training and evaluation complete!")
    print(f"Report and metrics saved in: {run_dir}")

if __name__ == "__main__":
    main()
