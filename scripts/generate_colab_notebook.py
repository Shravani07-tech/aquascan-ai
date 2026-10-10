import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": []
}

def add_markdown_cell(source_lines):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    })

def add_code_cell(source_lines):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    })

# ==============================================================================
# CELL 0: RUN ORDER & INSTRUCTIONS
# ==============================================================================
add_markdown_cell([
    "# 🛸 AquaScan AI — YOLOv8n 10-Epoch GPU Training Pipeline (Google Colab)",
    "",
    "This notebook executes the production **10-Epoch YOLOv8n GPU Training Pipeline** for **AquaScan AI**.",
    "All training outputs, model weights (`best.pt`, `last.pt`), metric curves, and visual predictions are **permanently saved directly to Google Drive**.",
    "",
    "---",
    "",
    "### 📋 RUN ORDER & EXECUTION INSTRUCTIONS:",
    "1. **Select GPU Runtime:** Go to `Runtime` > `Change runtime type` > Select `T4 GPU`.",
    "2. **Run Pipeline:** Select `Runtime` > `Run all` (or press `Ctrl + F9`).",
    "3. **Google Drive Authorization:** When prompted in **Step 1**, approve Google Drive access so results are saved permanently.",
    "4. **Dataset Source Options (Handled Automatically in Step 4):**",
    "   - **Option A (Recommended):** Upload `AquaScan_GPU_Training_Pack.zip` to Google Drive root or `/content/`.",
    "   - **Option B:** Have `dataset_clean/floating_waste_v10` already present in Google Drive or Colab.",
    "",
    "---"
])

# ==============================================================================
# CELL 1: CONFIGURATION CELL
# ==============================================================================
add_markdown_cell([
    "## Step 0: Central Pipeline Configuration",
    "Define all project paths, hyperparameters, and environment settings in a single configuration block."
])

add_code_cell([
    "# ==============================================================================",
    "# AQUASCAN AI — CENTRAL PIPELINE CONFIGURATION",
    "# ==============================================================================",
    "import os",
    "",
    "class AquaScanConfig:",
    "    # Hyperparameters (10 Epochs Production Config)",
    "    EPOCHS = 10",
    "    IMGSZ = 640",
    "    BATCH_SIZE = 16",
    "    WORKERS = 2",
    "    DEVICE = 0  # GPU 0",
    "    ",
    "    # Dataset Taxonomy",
    "    NUM_CLASSES = 5",
    "    CLASS_NAMES = ['bottle', 'can', 'carton', 'paper', 'plastic']",
    "    ",
    "    # Environment & Storage Paths",
    "    COLAB_CONTENT_DIR = '/content'",
    "    DRIVE_MOUNT_DIR = '/content/drive'",
    "    DRIVE_PROJECT_DIR = '/content/drive/MyDrive/AquaScan_Training'",
    "    RUN_NAME = 'yolov8n_full_10ep'",
    "    ",
    "    # Candidate Dataset Locations (Checked in priority order)",
    "    CANDIDATE_DATASET_DIRS = [",
    "        '/content/dataset_clean/floating_waste_v10',",
    "        '/content/floating_waste_v10',",
    "        '/content/drive/MyDrive/dataset_clean/floating_waste_v10',",
    "        '/content/drive/MyDrive/AquaScan_Dataset/floating_waste_v10'",
    "    ]",
    "    ",
    "    # Candidate ZIP Package Locations",
    "    CANDIDATE_ZIPS = [",
    "        '/content/drive/MyDrive/AquaScan_GPU_Training_Pack.zip',",
    "        '/content/AquaScan_GPU_Training_Pack.zip',",
    "        '/content/drive/MyDrive/floating_waste_v10.zip'",
    "    ]",

    "",
    "# Global Configuration Instance",
    "CFG = AquaScanConfig()",
    "print('✅ Central Configuration Loaded Successfully!')",
    "print(f'  - Training Epochs: {CFG.EPOCHS}')",
    "print(f'  - Target Classes ({CFG.NUM_CLASSES}): {CFG.CLASS_NAMES}')",
    "print(f'  - Output Directory: {CFG.DRIVE_PROJECT_DIR}')"
])

# ==============================================================================
# CELL 2: DRIVE SETUP
# ==============================================================================
add_markdown_cell([
    "## Step 1: Google Drive Mounting & Output Directory Setup",
    "Mount Google Drive to guarantee permanent storage for `best.pt`, metrics, and evaluation results."
])

add_code_cell([
    "import os",
    "from google.colab import drive",
    "",
    "print('[*] Mounting Google Drive...')",
    "try:",
    "    drive.mount(CFG.DRIVE_MOUNT_DIR, force_remount=False)",
    "    print('✅ Google Drive mounted successfully!')",
    "except Exception as e:",
    "    print(f'⚠️ Google Drive mount failed or already mounted: {e}')",
    "",
    "# Create permanent output project directories in Google Drive",
    "RUN_OUTPUT_DIR = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)",
    "WEIGHTS_DIR = os.path.join(RUN_OUTPUT_DIR, 'weights')",
    "PRED_DIR = os.path.join(RUN_OUTPUT_DIR, 'visual_predictions')",
    "",
    "os.makedirs(WEIGHTS_DIR, exist_ok=True)",
    "os.makedirs(PRED_DIR, exist_ok=True)",
    "",
    "print(f'✅ Permanent Drive Output Directory Ready: {RUN_OUTPUT_DIR}')",
    "print(f'  - Weights Subdirectory  : {WEIGHTS_DIR}')",
    "print(f'  - Predictions Subdirectory: {PRED_DIR}')"
])

# ==============================================================================
# CELL 3: GPU CHECK
# ==============================================================================
add_markdown_cell([
    "## Step 2: Verify NVIDIA GPU & CUDA Acceleration",
    "Ensure PyTorch CUDA acceleration is active. The pipeline requires an NVIDIA GPU (e.g. T4)."
])

add_code_cell([
    "import sys, torch",
    "",
    "print('[*] Verifying Hardware Acceleration...')",
    "# Execute nvidia-smi tool",
    "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv",
    "",
    "cuda_ok = torch.cuda.is_available()",
    "print(f'  - PyTorch CUDA Available: {cuda_ok}')",
    "if cuda_ok:",
    "    gpu_name = torch.cuda.get_device_name(CFG.DEVICE)",
    "    gpu_count = torch.cuda.device_count()",
    "    print(f'✅ GPU Accelerator Detected: {gpu_name} (Total Devices: {gpu_count})')",
    "else:",
    "    error_msg = (",
    "        '❌ ERROR: No GPU accelerator detected!\\n'",
    "        'Please enable GPU acceleration before running training:\\n'",
    "        '1. Go to Runtime -> Change runtime type\\n'",
    "        '2. Set Hardware accelerator to GPU (T4)\\n'",
    "        '3. Click Save and re-run all cells.'",
    "    )",
    "    print(error_msg)",
    "    raise RuntimeError('NVIDIA GPU Accelerator is required but missing.')"
])

# ==============================================================================
# CELL 4: PACKAGE SETUP
# ==============================================================================
add_markdown_cell([
    "## Step 3: Install & Verify Dependencies",
    "Install `ultralytics`, `opencv-python-headless`, `pyyaml`, `matplotlib`, and `pandas`."
])

add_code_cell([
    "print('[*] Installing required packages...')",
    "!pip install -q ultralytics opencv-python-headless pyyaml matplotlib pandas",
    "",
    "import cv2",
    "import yaml",
    "import matplotlib",
    "import pandas as pd",
    "import ultralytics",
    "from ultralytics import YOLO",
    "",
    "print('✅ Package Installation & Import Verification Complete:')",
    "print(f'  - Ultralytics Version: {ultralytics.__version__}')",
    "print(f'  - PyTorch Version    : {torch.__version__}')",
    "print(f'  - OpenCV Version     : {cv2.__version__}')",
    "print(f'  - PyYAML Version     : {yaml.__version__}')"
])

# ==============================================================================
# CELL 5: DATASET SETUP & EXTRACTION
# ==============================================================================
add_markdown_cell([
    "## Step 4: Dataset Resolution & Extraction",
    "Locate or extract the dataset cleanly from Google Drive or local workspace storage."
])

add_code_cell([
    "import os, sys, zipfile",
    "",
    "def resolve_dataset():",
    "    print('[*] Resolving Dataset Source...')",
    "    ",
    "    # 1. Check if dataset directory already exists",
    "    for cand_dir in CFG.CANDIDATE_DATASET_DIRS:",
    "        yaml_check = os.path.join(cand_dir, 'data.yaml')",
    "        if os.path.exists(cand_dir) and os.path.exists(yaml_check):",
    "            print(f'✅ Found existing extracted dataset at: {cand_dir}')",
    "            return cand_dir",
    "            ",
    "    # 2. Check if a dataset ZIP file is available for extraction",
    "    target_zip = None",
    "    for cand_zip in CFG.CANDIDATE_ZIPS:",
    "        if os.path.exists(cand_zip) and os.path.getsize(cand_zip) > 0:",
    "            target_zip = cand_zip",
    "            break",
    "            ",
    "    if target_zip:",
    "        print(f'[*] Found dataset ZIP package: {target_zip} ({os.path.getsize(target_zip) / (1024*1024):.2f} MB)')",
    "        print('[*] Validating ZIP archive integrity...')",
    "        if not zipfile.is_zipfile(target_zip):",
    "            raise ValueError(f'❌ File {target_zip} is not a valid ZIP archive.')",
    "            ",
    "        with zipfile.ZipFile(target_zip, 'r') as zf:",
    "            bad_file = zf.testzip()",
    "            if bad_file:",
    "                raise ValueError(f'❌ Corrupted file detected in ZIP: {bad_file}')",
    "            print('[*] Extracting ZIP package to /content/...')",
    "            zf.extractall(CFG.COLAB_CONTENT_DIR)",
    "        print('✅ ZIP extraction finished successfully!')",
    "        ",
    "        # Re-check candidate directories after extraction",
    "        for cand_dir in CFG.CANDIDATE_DATASET_DIRS:",
    "            yaml_check = os.path.join(cand_dir, 'data.yaml')",
    "            if os.path.exists(cand_dir) and os.path.exists(yaml_check):",
    "                print(f'✅ Dataset located after extraction at: {cand_dir}')",
    "                return cand_dir",
    "                ",
    "    # 3. Raise helpful error if dataset was not found",
    "    error_msg = (",
    "        '❌ ERROR: Could not locate AquaScan dataset or dataset ZIP!\\n'",
    "        'Please provide the dataset using one of the following methods:\\n'",
    "        f'1. Upload AquaScan_GPU_Training_Pack.zip to Google Drive root ({CFG.CANDIDATE_ZIPS[0]})\\n'",
    "        f'2. Upload AquaScan_GPU_Training_Pack.zip to /content/ ({CFG.CANDIDATE_ZIPS[1]})\\n'",
    "        f'3. Place extracted dataset folder at {CFG.CANDIDATE_DATASET_DIRS[0]}'",
    "    )",
    "    print(error_msg)",
    "    raise FileNotFoundError('Dataset files missing.')",

    "",
    "# Execute resolution",
    "ACTIVE_DATASET_DIR = resolve_dataset()",
    "print(f'✅ Active Dataset Path Set To: {ACTIVE_DATASET_DIR}')"
])

# ==============================================================================
# CELL 6: DATASET VALIDATION
# ==============================================================================
add_markdown_cell([
    "## Step 5: Dataset & YOLO Label Integrity Audit",
    "Validate `data.yaml` schema, image/label structure, and 5-field YOLO bounding box constraints."
])

add_code_cell([
    "import os, glob, yaml",
    "",
    "def audit_dataset_integrity(dataset_dir):",
    "    print(f'[*] Auditing Dataset Integrity at: {dataset_dir}')",
    "    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')",
    "    ",
    "    if not os.path.exists(data_yaml_path):",
    "        raise FileNotFoundError(f'Missing data.yaml at {data_yaml_path}')",
    "        ",
    "    # Overwrite data.yaml with verified absolute paths for Colab environment",
    "    yaml_payload = {",
    "        'path': dataset_dir,",
    "        'train': 'train/images',",
    "        'val': 'valid/images',",
    "        'test': 'test/images',",
    "        'nc': CFG.NUM_CLASSES,",
    "        'names': CFG.CLASS_NAMES",
    "    }",
    "    with open(data_yaml_path, 'w') as f:",
    "        yaml.dump(yaml_payload, f)",
    "    print(f'✅ Updated data.yaml configuration: {data_yaml_path}')",
    "    ",
    "    splits = ['train', 'valid', 'test']",
    "    total_images, total_labels, total_boxes, invalid_boxes = 0, 0, 0, 0",
    "    class_counts = {name: 0 for name in CFG.CLASS_NAMES}",
    "    ",
    "    for split in splits:",
    "        img_dir = os.path.join(dataset_dir, split, 'images')",
    "        lbl_dir = os.path.join(dataset_dir, split, 'labels')",
    "        ",
    "        if not os.path.exists(img_dir):",
    "            raise FileNotFoundError(f'Missing image directory: {img_dir}')",
    "        if not os.path.exists(lbl_dir):",
    "            raise FileNotFoundError(f'Missing label directory: {lbl_dir}')",
    "            ",
    "        imgs = glob.glob(os.path.join(img_dir, '*'))",
    "        lbls = glob.glob(os.path.join(lbl_dir, '*.txt'))",
    "        total_images += len(imgs)",
    "        total_labels += len(lbls)",
    "        ",
    "        for lf in lbls:",
    "            if os.path.getsize(lf) == 0: continue",
    "            with open(lf, 'r') as f:",
    "                for line in f:",
    "                    parts = line.strip().split()",
    "                    if not parts: continue",
    "                    total_boxes += 1",
    "                    if len(parts) != 5:",
    "                        invalid_boxes += 1",
    "                        continue",
    "                    cls_id = int(parts[0])",
    "                    if 0 <= cls_id < CFG.NUM_CLASSES:",
    "                        class_counts[CFG.CLASS_NAMES[cls_id]] += 1",
    "                    else:",
    "                        invalid_boxes += 1",
    "                        ",
    "        print(f'  [+] Split {split.upper():5s}: {len(imgs):5d} images | {len(lbls):5d} label files')",
    "        ",
    "    print('\\n==================================================')",
    "    print('         DATASET AUDIT SUMMARY')",
    "    print('==================================================')",
    "    print(f'  Total Images Found    : {total_images}')",
    "    print(f'  Total Label Files     : {total_labels}')",
    "    print(f'  Total Bounding Boxes  : {total_boxes}')",
    "    print(f'  Invalid 5-Field Boxes : {invalid_boxes}')",
    "    print(f'  Class Breakdown       : {class_counts}')",
    "    print('==================================================')",
    "    ",
    "    if invalid_boxes > 0:",
    "        raise ValueError(f'Found {invalid_boxes} invalid label entries.')",
    "    if total_images == 0:",
    "        raise ValueError('Dataset contains 0 images.')",
    "        ",
    "    print('✅ Dataset Validation 100% PASSED!')",
    "    return data_yaml_path",

    "",
    "DATA_YAML_PATH = audit_dataset_integrity(ACTIVE_DATASET_DIR)"
])

# ==============================================================================
# CELL 7: TRAINING
# ==============================================================================
add_markdown_cell([
    "## Step 6: 10-Epoch YOLOv8n GPU Training",
    "Execute 10-epoch GPU training. Model weights (`best.pt`, `last.pt`) and metric curves are saved directly to Google Drive."
])

add_code_cell([
    "import time, os",
    "from ultralytics import YOLO",
    "",
    "print('=' * 70)",
    "print(f'  STARTING {CFG.EPOCHS}-EPOCH YOLOv8n GPU TRAINING')",
    "print('=' * 70)",
    "print(f'  Data YAML Path : {DATA_YAML_PATH}')",
    "print(f'  Target Epochs  : {CFG.EPOCHS}')",
    "print(f'  Image Size     : {CFG.IMGSZ}')",
    "print(f'  Batch Size     : {CFG.BATCH_SIZE}')",
    "print(f'  Device         : GPU {CFG.DEVICE}')",
    "print(f'  Output Drive   : {CFG.DRIVE_PROJECT_DIR}')",
    "print('=' * 70)",
    "",
    "# Load baseline pretrained YOLOv8n weights",
    "model = YOLO('yolov8n.pt')",
    "",
    "start_time = time.time()",
    "",
    "# Execute Ultralytics training with direct Drive output",
    "training_results = model.train(",
    "    data=DATA_YAML_PATH,",
    "    epochs=CFG.EPOCHS,",
    "    imgsz=CFG.IMGSZ,",
    "    batch=CFG.BATCH_SIZE,",
    "    workers=CFG.WORKERS,",
    "    device=CFG.DEVICE,",
    "    project=CFG.DRIVE_PROJECT_DIR,",
    "    name=CFG.RUN_NAME,",
    "    exist_ok=True,",
    "    save=True,",
    "    plots=True,",
    "    verbose=True",
    ")",
    "",
    "elapsed_sec = time.time() - start_time",
    "print('\\n' + '=' * 70)",
    "print(f'✅ GPU TRAINING COMPLETE in {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} minutes)!')",
    "print('=' * 70)",
    "",
    "# Verify best.pt physical existence in Google Drive",
    "EXPECTED_BEST_WEIGHTS = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'best.pt')",
    "EXPECTED_LAST_WEIGHTS = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'last.pt')",
    "",
    "if os.path.exists(EXPECTED_BEST_WEIGHTS) and os.path.getsize(EXPECTED_BEST_WEIGHTS) > 0:",
    "    print(f'✅ Physical Weight Check PASSED! best.pt saved at: {EXPECTED_BEST_WEIGHTS} ({os.path.getsize(EXPECTED_BEST_WEIGHTS)} bytes)')",
    "else:",
    "    raise FileNotFoundError(f'❌ Training completed but best.pt was not found at {EXPECTED_BEST_WEIGHTS}')"
])

# ==============================================================================
# CELL 8: STEP 6 VALIDATION EVALUATION
# ==============================================================================
add_markdown_cell([
    "## Step 7: Validation Set Evaluation (`val` split)",
    "Load verified `best.pt` from Google Drive and evaluate precision, recall, mAP50, and mAP50-95 on the 638 validation images."
])

add_code_cell([
    "import os",
    "from ultralytics import YOLO",
    "",
    "print('[*] Loading verified best.pt model from Google Drive...')",
    "best_weights_path = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'best.pt')",
    "if not os.path.exists(best_weights_path):",
    "    raise FileNotFoundError(f'Missing best.pt at {best_weights_path}')",
    "    ",
    "best_model = YOLO(best_weights_path)",
    "",
    "print('\\n==================================================')",
    "print('    VALIDATION SET PERFORMANCE EVALUATION (val)')",
    "print('==================================================')",
    "val_metrics = best_model.val(data=DATA_YAML_PATH, split='val', device=CFG.DEVICE, verbose=True)",
    "",
    "val_dict = val_metrics.results_dict",
    "precision_val = val_dict.get('metrics/precision(B)', 0.0)",
    "recall_val = val_dict.get('metrics/recall(B)', 0.0)",
    "map50_val = val_dict.get('metrics/mAP50(B)', 0.0)",
    "map50_95_val = val_dict.get('metrics/mAP50-95(B)', 0.0)",
    "",
    "print('\\n--- OVERALL VALIDATION METRICS ---')",
    "print(f'  Validation Precision (B) : {precision_val * 100:.2f}% ({precision_val:.4f})')",
    "print(f'  Validation Recall (B)    : {recall_val * 100:.2f}% ({recall_val:.4f})')",
    "print(f'  Validation mAP50 (B)     : {map50_val * 100:.2f}% ({map50_val:.4f})')",
    "print(f'  Validation mAP50-95 (B)  : {map50_95_val * 100:.2f}% ({map50_95_val:.4f})')",
    "",
    "print('\\n--- PER-CLASS VALIDATION METRICS ---')",
    "if hasattr(val_metrics, 'box') and hasattr(val_metrics.box, 'p'):",
    "    p = val_metrics.box.p",
    "    r = val_metrics.box.r",
    "    ap50 = val_metrics.box.ap50",
    "    ap = val_metrics.box.ap",
    "    for i, cname in enumerate(CFG.CLASS_NAMES):",
    "        if i < len(p):",
    "            print(f'  - Class {i} ({cname:10s}): Precision={p[i]:.4f}, Recall={r[i]:.4f}, mAP50={ap50[i]:.4f}, mAP50-95={ap[i]:.4f}')"
])

# ==============================================================================
# CELL 9: STEP 7 TEST EVALUATION
# ==============================================================================
add_markdown_cell([
    "## Step 8: Test Set Generalization Evaluation (`test` split)",
    "Evaluate model generalization performance on the 319 unseen test set images."
])

add_code_cell([
    "import os",
    "from ultralytics import YOLO",
    "",
    "print('[*] Loading verified best.pt model from Google Drive for Test Evaluation...')",
    "best_weights_path = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'best.pt')",
    "best_model = YOLO(best_weights_path)",
    "",
    "print('\\n==================================================')",
    "print('       TEST SET GENERALIZATION EVALUATION (test)')",
    "print('==================================================')",
    "test_metrics = best_model.val(data=DATA_YAML_PATH, split='test', device=CFG.DEVICE, verbose=True)",
    "",
    "test_dict = test_metrics.results_dict",
    "precision_test = test_dict.get('metrics/precision(B)', 0.0)",
    "recall_test = test_dict.get('metrics/recall(B)', 0.0)",
    "map50_test = test_dict.get('metrics/mAP50(B)', 0.0)",
    "map50_95_test = test_dict.get('metrics/mAP50-95(B)', 0.0)",
    "",
    "print('\\n--- OVERALL TEST SET METRICS ---')",
    "print(f'  Test Precision (B) : {precision_test * 100:.2f}% ({precision_test:.4f})')",
    "print(f'  Test Recall (B)    : {recall_test * 100:.2f}% ({recall_test:.4f})')",
    "print(f'  Test mAP50 (B)     : {map50_test * 100:.2f}% ({map50_test:.4f})')",
    "print(f'  Test mAP50-95 (B)  : {map50_95_test * 100:.2f}% ({map50_95_test:.4f})')",
    "",
    "print('\\n--- PER-CLASS TEST METRICS ---')",
    "if hasattr(test_metrics, 'box') and hasattr(test_metrics.box, 'p'):",
    "    p = test_metrics.box.p",
    "    r = test_metrics.box.r",
    "    ap50 = test_metrics.box.ap50",
    "    ap = test_metrics.box.ap",
    "    for i, cname in enumerate(CFG.CLASS_NAMES):",
    "        if i < len(p):",
    "            print(f'  - Class {i} ({cname:10s}): Precision={p[i]:.4f}, Recall={r[i]:.4f}, mAP50={ap50[i]:.4f}, mAP50-95={ap[i]:.4f}')"
])

# ==============================================================================
# CELL 10: STEP 8 INFERENCE BENCHMARK & PREDICTIONS
# ==============================================================================
add_markdown_cell([
    "## Step 9: GPU Inference Latency Benchmark & Visual Predictions",
    "Benchmark GPU inference latency (ms per frame & FPS) and save visual bounding box predictions permanently to Google Drive."
])

add_code_cell([
    "import os, glob, cv2",
    "from ultralytics import YOLO",
    "",
    "best_weights_path = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'best.pt')",
    "best_model = YOLO(best_weights_path)",
    "",
    "test_img_dir = os.path.join(ACTIVE_DATASET_DIR, 'test', 'images')",
    "test_imgs = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))[:20]",
    "",
    "pred_out_dir = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'visual_predictions')",
    "os.makedirs(pred_out_dir, exist_ok=True)",
    "",
    "print(f'[*] Benchmarking GPU inference across {len(test_imgs)} test images...')",
    "",
    "# GPU Warmup Run",
    "if test_imgs:",
    "    best_model.predict(source=test_imgs[0], device=CFG.DEVICE, verbose=False)",
    "",
    "pre_times, inf_times, post_times = [], [], []",
    "saved_paths = []",
    "",
    "for img_p in test_imgs:",
    "    fname = os.path.basename(img_p)",
    "    results = best_model.predict(source=img_p, device=CFG.DEVICE, conf=0.10, verbose=False)",
    "    for r in results:",
    "        sp = r.speed",
    "        pre_times.append(sp.get('preprocess', 0.0))",
    "        inf_times.append(sp.get('inference', 0.0))",
    "        post_times.append(sp.get('postprocess', 0.0))",
    "        ",
    "        # Save visual prediction plot to Google Drive",
    "        annotated_bgr = r.plot()",
    "        save_path = os.path.join(pred_out_dir, f'pred_{fname}')",
    "        cv2.imwrite(save_path, annotated_bgr)",
    "        saved_paths.append(save_path)",
    "",
    "avg_pre = sum(pre_times) / len(pre_times) if pre_times else 0.0",
    "avg_inf = sum(inf_times) / len(inf_times) if inf_times else 0.0",
    "avg_post = sum(post_times) / len(post_times) if post_times else 0.0",
    "avg_total = avg_pre + avg_inf + avg_post",
    "fps = 1000.0 / avg_total if avg_total > 0 else 0.0",
    "",
    "print('\\n==================================================')",
    "print('     GPU INFERENCE LATENCY BENCHMARK RESULTS')",
    "print('==================================================')",
    "print(f'  Preprocess Latency  : {avg_pre:.2f} ms')",
    "print(f'  Inference Latency   : {avg_inf:.2f} ms')",
    "print(f'  Postprocess Latency : {avg_post:.2f} ms')",
    "print(f'  Total Frame Latency : {avg_total:.2f} ms')",
    "print(f'  Throughput (FPS)    : {fps:.2f} FPS')",
    "print(f'  Visual Predictions  : Saved {len(saved_paths)} images to Drive ({pred_out_dir})')",
    "print('==================================================')"
])

# ==============================================================================
# CELL 11: FINAL VERIFICATION
# ==============================================================================
add_markdown_cell([
    "## Step 10: Final Deliverables Audit & Google Drive Verification",
    "Verify that all final deliverables (`best.pt`, `last.pt`, `results.csv`, training curves, confusion matrix, predictions) exist in Google Drive."
])

add_code_cell([
    "import os",
    "",
    "run_dir = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)",
    "weights_dir = os.path.join(run_dir, 'weights')",
    "preds_dir = os.path.join(run_dir, 'visual_predictions')",
    "",
    "expected_files = [",
    "    ('best.pt Model Weights', os.path.join(weights_dir, 'best.pt')),",
    "    ('last.pt Model Weights', os.path.join(weights_dir, 'last.pt')),",
    "    ('Results CSV Metrics', os.path.join(run_dir, 'results.csv')),",
    "    ('Confusion Matrix Plot', os.path.join(run_dir, 'confusion_matrix.png')),",
    "    ('Training Curves Plot', os.path.join(run_dir, 'results.png')),",
    "    ('Visual Predictions Dir', preds_dir)",
    "]",
    "",
    "print('=' * 75)",
    "print('           FINAL AQUASCAN AI DELIVERABLES VERIFICATION')",
    "print('=' * 75)",
    "print(f'{\"Deliverable Name\":<25} | {\"Status\":<8} | {\"Size (MB/KB)\":<12} | Path')",
    "print('-' * 75)",
    "",
    "all_ok = True",
    "for label, path in expected_files:",
    "    exists = os.path.exists(path)",
    "    if exists:",
    "        if os.path.isdir(path):",
    "            cnt = len(os.listdir(path))",
    "            size_str = f'{cnt} files'",
    "        else:",
    "            sz = os.path.getsize(path)",
    "            size_str = f'{sz/(1024*1024):.2f} MB' if sz > 1024*1024 else f'{sz/1024:.1f} KB'",
    "        status_str = '✅ EXISTS'",
    "    else:",
    "        status_str = '❌ MISSING'",
    "        size_str = 'N/A'",
    "        all_ok = False",
    "    print(f'{label:<25} | {status_str:<8} | {size_str:<12} | {path}')",
    "",
    "print('=' * 75)",
    "if all_ok:",
    "    print('🚀 SUCCESS: ALL DELIVERABLES ARE PERMANENTLY STORED IN GOOGLE DRIVE!')",
    "    print(f'   Target Model Path: {os.path.join(weights_dir, \"best.pt\")}')",
    "else:",
    "    print('⚠️ WARNING: Some expected deliverables are missing. Please review training logs.')",
    "print('=' * 75)"
])

# Save notebook to workspace root as AquaScan_YOLOv8n_GPU_Training.ipynb
out_path = os.path.abspath("AquaScan_YOLOv8n_GPU_Training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated new clean notebook at: {out_path} ({os.path.getsize(out_path)} bytes)")
