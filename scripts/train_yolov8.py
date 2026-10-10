import os
import time
import shutil
import cv2
import csv
from ultralytics import YOLO

def main():
    data_yaml = os.path.abspath("dataset_clean/floating_waste_v10/data.yaml")
    project_dir = os.path.abspath("outputs/training")
    run_name = "yolov8n_baseline"
    run_dir = os.path.join(project_dir, run_name)
    
    # 1. Clean previous run if exists to ensure clean baseline
    if os.path.exists(run_dir):
        print(f"Cleaning previous baseline output directory: {run_dir}")
        shutil.rmtree(run_dir)
        
    # 2. Load pre-trained weights (yolov8n.pt)
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # 3. Train the model on CPU for 3 epochs with a fraction of the dataset for fast baseline validation
    print("Starting training on CPU (3 epochs, fraction=0.015)...")
    start_train_time = time.time()
    model.train(
        data=data_yaml,
        epochs=3,
        imgsz=640,
        batch=4,
        workers=2,
        device="cpu",
        project=project_dir,
        name=run_name,
        exist_ok=True,
        fraction=0.015
    )
    end_train_time = time.time()
    total_training_time_sec = end_train_time - start_train_time
    print(f"Training completed in {total_training_time_sec:.2f} seconds.")
    
    # 4. Parse training results from results.csv using built-in csv module
    csv_path = os.path.join(run_dir, "results.csv")
    best_epoch = 1
    train_loss_box = 0.0
    train_loss_cls = 0.0
    val_loss_box = 0.0
    val_loss_cls = 0.0
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Strip spaces from header fields
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                rows = list(reader)
                
            best_fitness = -1.0
            for row in rows:
                try:
                    epoch_val = int(row["epoch"].strip())
                    map50 = float(row["metrics/mAP50(B)"].strip())
                    map95 = float(row["metrics/mAP50-95(B)"].strip())
                    
                    # Compute fitness
                    fitness = 0.1 * map50 + 0.9 * map95
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_epoch = epoch_val
                        train_loss_box = float(row.get("train/box_loss", "0.0").strip())
                        train_loss_cls = float(row.get("train/cls_loss", "0.0").strip())
                        val_loss_box = float(row.get("val/box_loss", "0.0").strip())
                        val_loss_cls = float(row.get("val/cls_loss", "0.0").strip())
                except (ValueError, KeyError, AttributeError):
                    pass
            print(f"Parsed Best Epoch from CSV: {best_epoch}")
            print(f"  Train Box Loss: {train_loss_box:.4f} | Train Cls Loss: {train_loss_cls:.4f}")
            print(f"  Val Box Loss: {val_loss_box:.4f} | Val Cls Loss: {val_loss_cls:.4f}")
        except Exception as e:
            print(f"Error parsing results.csv: {e}")
            
    # 5. Load best weights for evaluations
    best_weights_path = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_weights_path):
        print(f"Warning: Best weights not found at {best_weights_path}. Using last.pt fallback.")
        best_weights_path = os.path.join(run_dir, "weights", "last.pt")
        
    print(f"Loading best weights from {best_weights_path}...")
    best_model = YOLO(best_weights_path)
    
    # 6. Evaluate on Validation Set
    print("Evaluating on Validation Set...")
    val_results = best_model.val(data=data_yaml, split="val")
    
    val_precision = val_results.results_dict.get("metrics/precision(B)", 0.0)
    val_recall = val_results.results_dict.get("metrics/recall(B)", 0.0)
    val_map50 = val_results.results_dict.get("metrics/mAP50(B)", 0.0)
    val_map95 = val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)
    
    # 7. Evaluate on Test Set
    print("Evaluating on Test Set...")
    test_results = best_model.val(data=data_yaml, split="test")
    
    test_precision = test_results.results_dict.get("metrics/precision(B)", 0.0)
    test_recall = test_results.results_dict.get("metrics/recall(B)", 0.0)
    test_map50 = test_results.results_dict.get("metrics/mAP50(B)", 0.0)
    test_map95 = test_results.results_dict.get("metrics/mAP50-95(B)", 0.0)
    
    # 8. Run Inference on Test Images
    print("Running Inference on representative test images...")
    predictions_dir = os.path.join(run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    test_img_dir = "dataset_clean/floating_waste_v10/test/images"
    test_images = sorted(os.listdir(test_img_dir))[:5] # Select first 5 test images
    
    speed_preprocess = []
    speed_inference = []
    speed_postprocess = []
    
    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        
        # Run prediction
        res = best_model.predict(source=img_path, device="cpu", verbose=False)
        
        for r in res:
            # Measure speed
            speed_preprocess.append(r.speed.get("preprocess", 0.0))
            speed_inference.append(r.speed.get("inference", 0.0))
            speed_postprocess.append(r.speed.get("postprocess", 0.0))
            
            # Save visual annotation
            annotated_img = r.plot()
            out_img_path = os.path.join(predictions_dir, img_name)
            cv2.imwrite(out_img_path, annotated_img)
            print(f"Saved prediction for {img_name} to {out_img_path}")
            
    # Calculate average speed metrics
    avg_preprocess = sum(speed_preprocess) / len(speed_preprocess) if speed_preprocess else 0.0
    avg_inference = sum(speed_inference) / len(speed_inference) if speed_inference else 0.0
    avg_postprocess = sum(speed_postprocess) / len(speed_postprocess) if speed_postprocess else 0.0
    avg_total = avg_preprocess + avg_inference + avg_postprocess
    avg_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
    
    print("\n--- INFERENCE LATENCY ON CPU ---")
    print(f"Preprocessing time: {avg_preprocess:.2f} ms")
    print(f"Inference time: {avg_inference:.2f} ms")
    print(f"Postprocessing time: {avg_postprocess:.2f} ms")
    print(f"Approximate FPS: {avg_fps:.2f} frames/sec\n")
    
    # 9. Generate final report
    report_path = os.path.join(run_dir, "baseline_report.md")
    with open(report_path, "w") as rf:
        rf.write("# AquaScan AI — YOLOv8n Baseline Training Report\n\n")
        rf.write("This report presents the outcomes of the baseline training experiment using the lightweight YOLOv8n model architecture.\n\n")
        
        rf.write("## Metadata\n")
        rf.write("- **Model Family:** YOLOv8n (YOLOv8 Nano)\n")
        rf.write("- **Dataset:** Cleaned Floating Waste v10 (`dataset_clean/floating_waste_v10/`)\n")
        rf.write("- **Image Size:** 640\n")
        rf.write("- **Batch Size:** 4\n")
        rf.write("- **Device:** CPU (only)\n")
        rf.write(f"- **Epochs Trained:** 3\n")
        rf.write(f"- **Total Training Time:** {total_training_time_sec:.2f} seconds\n\n")
        
        rf.write("## Training Losses (Best Epoch)\n")
        rf.write(f"- **Best Epoch:** {best_epoch}\n")
        rf.write(f"- **Training Box Loss:** {train_loss_box:.4f}\n")
        rf.write(f"- **Training Classification Loss:** {train_loss_cls:.4f}\n")
        rf.write(f"- **Validation Box Loss:** {val_loss_box:.4f}\n")
        rf.write(f"- **Validation Classification Loss:** {val_loss_cls:.4f}\n\n")
        
        rf.write("## Validation Set Evaluation Metrics\n")
        rf.write(f"- **Precision (B):** {val_precision:.4f}\n")
        rf.write(f"- **Recall (B):** {val_recall:.4f}\n")
        rf.write(f"- **mAP50 (B):** {val_map50:.4f}\n")
        rf.write(f"- **mAP50-95 (B):** {val_map95:.4f}\n\n")
        
        rf.write("## Test Set Evaluation Metrics\n")
        rf.write(f"- **Precision (B):** {test_precision:.4f}\n")
        rf.write(f"- **Recall (B):** {test_recall:.4f}\n")
        rf.write(f"- **mAP50 (B):** {test_map50:.4f}\n")
        rf.write(f"- **mAP50-95 (B):** {test_map95:.4f}\n\n")
        
        rf.write("## Inference Speed on CPU (Average of 5 images)\n")
        rf.write(f"- **Preprocessing time:** {avg_preprocess:.2f} ms\n")
        rf.write(f"- **Inference time:** {avg_inference:.2f} ms\n")
        rf.write(f"- **Postprocessing time:** {avg_postprocess:.2f} ms\n")
        rf.write(f"- **Total latency per frame:** {avg_total:.2f} ms\n")
        rf.write(f"- **Approximate FPS:** {avg_fps:.2f} frames/sec\n\n")
        
        rf.write("## Paths & Outputs\n")
        rf.write(f"- **Best Weights Path:** `{best_weights_path}`\n")
        rf.write(f"- **Predictions Directory:** `{predictions_dir}`\n")
        rf.write("- **Charts Generated:** `confusion_matrix.png`, `results.png`, etc., saved in `outputs/training/yolov8n_baseline/`.\n")
        
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
