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
# CELL 0: RUN ORDER & USER GUIDE
# ==============================================================================
add_markdown_cell([
    "# 🛸 AquaScan AI — YOLOv8n Final 10-Epoch GPU Training Notebook",
    "",
    "This notebook executes the **FINAL 10-EPOCH YOLOv8n GPU TRAINING RUN** for **AquaScan AI**.",
    "All training outputs, model weights (`best.pt`, `last.pt`), evaluation metrics, visual predictions, and final ZIP artifacts are **permanently saved directly to Google Drive**.",
    "",
    "---",
    "",
    "### 📋 RUN ORDER & EXECUTION INSTRUCTIONS:",
    "1. **Select GPU Accelerator:** In Google Colab, click `Runtime` > `Change runtime type` > Select `T4 GPU` > Click **Save**.",
    "2. **Provide Dataset (Auto-Detected in Step 4):**",
    "   - **Option A:** Upload `AquaScan_GPU_Training_Pack.zip` directly to Colab workspace (`/content/`).",
    "   - **Option B:** Place `AquaScan_GPU_Training_Pack.zip` in Google Drive root (`/content/drive/MyDrive/`).",
    "   - **Option C:** Have `floating_waste_v10` dataset directory already in Google Drive.",
    "3. **Run All Cells:** Click `Runtime` > `Run all` (or press `Ctrl + F9`).",
    "4. **Authorize Google Drive:** Click \"Connect to Google Drive\" when prompted in **Step 1**.",
    "",
    "---"
])

# ==============================================================================
# CELL 1: CENTRAL PIPELINE CONFIGURATION
# ==============================================================================
add_markdown_cell([
    "## Step 0: Central Pipeline Configuration",
    "Define all project hyper-parameters, environment settings, and Google Drive target paths in one central block."
])

add_code_cell([
    "# ==============================================================================",
    "# AQUASCAN AI — FINAL GPU TRAINING CONFIGURATION",
    "# ==============================================================================",
    "import os",
    "",
    "class AquaScanFinalConfig:",
    "    # Final 10-Epoch Hyperparameters",
    "    EPOCHS = 10",
    "    IMGSZ = 640",
    "    BATCH_SIZE = 16",
    "    WORKERS = 2",
    "    DEVICE = 0  # GPU Device 0",
    "    FRACTION = 1.0  # 100% of dataset (Full Training)",
    "    ",
    "    # Dataset Taxonomy (5 Marine Waste Classes)",
    "    NUM_CLASSES = 5",
    "    CLASS_NAMES = ['bottle', 'can', 'carton', 'paper', 'plastic']",
    "    ",
    "    # Environment & Storage Paths",
    "    COLAB_CONTENT_DIR = '/content'",
    "    DRIVE_MOUNT_DIR = '/content/drive'",
    "    DRIVE_PROJECT_DIR = '/content/drive/MyDrive/AquaScan_Training'",
    "    RUN_NAME = 'yolov8n_FINAL_10ep'",
    "    FINAL_ZIP_NAME = 'AquaScan_FINAL_10ep_Artifacts.zip'",
    "    ",
    "    # Candidate Dataset Folder Locations (Checked in priority order)",
    "    CANDIDATE_DATASET_DIRS = [",
    "        '/content/dataset_clean/floating_waste_v10',",
    "        '/content/floating_waste_v10',",
    "        '/content/drive/MyDrive/dataset_clean/floating_waste_v10',",
    "        '/content/drive/MyDrive/AquaScan_Dataset/floating_waste_v10'",
    "    ]",
    "    ",
    "    # Candidate Dataset ZIP Package Locations",
    "    CANDIDATE_ZIPS = [",
    "        '/content/AquaScan_GPU_Training_Pack.zip',",
    "        '/content/drive/MyDrive/AquaScan_GPU_Training_Pack.zip',",
    "        '/content/drive/MyDrive/floating_waste_v10.zip'",
    "    ]",

    "",
    "# Global Configuration Instance",
    "CFG = AquaScanFinalConfig()",
    "print('✅ AquaScan Final GPU Pipeline Configuration Loaded Successfully!')",
    "print(f'  - Training Epochs : {CFG.EPOCHS}')",
    "print(f'  - Dataset Fraction: {CFG.FRACTION * 100:.0f}% (Full Dataset)')",
    "print(f'  - Target Classes ({CFG.NUM_CLASSES}): {CFG.CLASS_NAMES}')",
    "print(f'  - Output Drive Dir: {os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)}')"
])

# ==============================================================================
# CELL 2: GOOGLE DRIVE MOUNTING
# ==============================================================================
add_markdown_cell([
    "## Step 1: Google Drive Mounting & Permanent Output Setup",
    "Mount Google Drive automatically and create permanent output directories for `best.pt`, plots, and final artifacts."
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
    "    print(f'⚠️ Google Drive mount check: {e}')",
    "",
    "# Create permanent output project directories in Google Drive",
    "RUN_OUTPUT_DIR = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)",
    "WEIGHTS_DIR = os.path.join(RUN_OUTPUT_DIR, 'weights')",
    "PRED_DIR = os.path.join(RUN_OUTPUT_DIR, 'visual_predictions')",
    "",
    "os.makedirs(WEIGHTS_DIR, exist_ok=True)",
    "os.makedirs(PRED_DIR, exist_ok=True)",
    "",
    "print(f'✅ Permanent Drive Target Directory Ready: {RUN_OUTPUT_DIR}')",
    "print(f'  - Weights Dir      : {WEIGHTS_DIR}')",
    "print(f'  - Predictions Dir  : {PRED_DIR}')"
])

# ==============================================================================
# CELL 3: GPU HARDWARE VERIFICATION
# ==============================================================================
add_markdown_cell([
    "## Step 2: Verify NVIDIA GPU Accelerator & CUDA Availability",
    "Strict hardware check. Ensures PyTorch CUDA is active on an NVIDIA GPU (e.g. Tesla T4). Hard stops if GPU is missing."
])

add_code_cell([
    "import sys, torch",
    "",
    "print('[*] Verifying NVIDIA GPU Accelerator & CUDA Status...')",
    "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv",
    "",
    "cuda_ok = torch.cuda.is_available()",
    "print(f'  - PyTorch CUDA Available: {cuda_ok}')",
    "if cuda_ok:",
    "    gpu_name = torch.cuda.get_device_name(CFG.DEVICE)",
    "    gpu_count = torch.cuda.device_count()",
    "    cuda_ver = torch.version.cuda",
    "    print(f'✅ NVIDIA GPU Verified: {gpu_name} (Devices: {gpu_count}, CUDA Version: {cuda_ver})')",
    "else:",
    "    error_msg = (",
    "        '❌ CRITICAL ERROR: NVIDIA GPU Accelerator is NOT detected!\\n'",
    "        'This pipeline requires GPU acceleration (Tesla T4).\\n'",
    "        'To enable GPU in Google Colab:\\n'",
    "        '  1. Go to Runtime -> Change runtime type\\n'",
    "        '  2. Set Hardware accelerator to GPU (T4)\\n'",
    "        '  3. Save and re-run all cells.'",
    "    )",
    "    print(error_msg)",
    "    raise RuntimeError('NVIDIA GPU Accelerator is required but missing. CPU fallback is strictly disallowed.')"
])

# ==============================================================================
# CELL 4: PACKAGE SETUP
# ==============================================================================
add_markdown_cell([
    "## Step 3: Install & Verify Required Dependencies",
    "Install `ultralytics`, `opencv-python-headless`, `pyyaml`, `matplotlib`, `pandas`, and verify imports."
])

add_code_cell([
    "print('[*] Installing required dependencies...')",
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
    "print(f'  - Ultralytics Version : {ultralytics.__version__}')",
    "print(f'  - PyTorch Version     : {torch.__version__}')",
    "print(f'  - OpenCV Version      : {cv2.__version__}')",
    "print(f'  - PyYAML Version      : {yaml.__version__}')"
])

# ==============================================================================
# CELL 5: DATASET SETUP & EXTRACTION
# ==============================================================================
add_markdown_cell([
    "## Step 4: Robust Dataset Resolution & Archive Integrity Check",
    "Automatically detects dataset Option A (uploaded ZIP), Option B (Drive ZIP), or Option C (extracted folder). Validates archive integrity before safe extraction."
])

add_code_cell([
    "import os, sys, zipfile",
    "",
    "def resolve_and_extract_dataset():",
    "    print('[*] Auto-Detecting Dataset Source (Options A, B, C)...')",
    "    ",
    "    # Option C: Check if extracted dataset directory already exists",
    "    for cand_dir in CFG.CANDIDATE_DATASET_DIRS:",
    "        yaml_check = os.path.join(cand_dir, 'data.yaml')",
    "        if os.path.exists(cand_dir) and os.path.exists(yaml_check):",
    "            print(f'✅ Found existing extracted dataset at: {cand_dir}')",
    "            return cand_dir",
    "            ",
    "    # Options A & B: Check for dataset ZIP file",
    "    target_zip = None",
    "    for cand_zip in CFG.CANDIDATE_ZIPS:",
    "        if os.path.exists(cand_zip) and os.path.getsize(cand_zip) > 0:",
    "            target_zip = cand_zip",
    "            break",
    "            ",
    "    if target_zip:",
    "        size_mb = os.path.getsize(target_zip) / (1024 * 1024)",
    "        print(f'[*] Found Dataset ZIP Package: {target_zip} ({size_mb:.2f} MB)')",
    "        print('[*] Checking ZIP Archive Integrity...')",
    "        ",
    "        if not zipfile.is_zipfile(target_zip):",
    "            raise ValueError(f'❌ Archive {target_zip} is not a valid ZIP file.')",
    "            ",
    "        with zipfile.ZipFile(target_zip, 'r') as zf:",
    "            corrupt_file = zf.testzip()",
    "            if corrupt_file:",
    "                raise ValueError(f'❌ Corrupted file detected inside ZIP archive: {corrupt_file}')",
    "            print('[*] Extracting ZIP package to /content/...')",
    "            zf.extractall(CFG.COLAB_CONTENT_DIR)",
    "        print('✅ ZIP Package Extracted Successfully!')",
    "        ",
    "        # Re-check dataset directory post-extraction",
    "        for cand_dir in CFG.CANDIDATE_DATASET_DIRS:",
    "            yaml_check = os.path.join(cand_dir, 'data.yaml')",
    "            if os.path.exists(cand_dir) and os.path.exists(yaml_check):",
    "                print(f'✅ Extracted Dataset Located at: {cand_dir}')",
    "                return cand_dir",
    "                ",
    "    # Clear error message if no dataset source is found",
    "    error_msg = (",
    "        '❌ CRITICAL ERROR: Could not locate AquaScan dataset or ZIP package!\\n'",
    "        'Please provide the dataset using one of the following methods:\\n'",
    "        f'  - Option A: Upload AquaScan_GPU_Training_Pack.zip to /content/ ({CFG.CANDIDATE_ZIPS[0]})\\n'",
    "        f'  - Option B: Place AquaScan_GPU_Training_Pack.zip in Google Drive root ({CFG.CANDIDATE_ZIPS[1]})\\n'",
    "        f'  - Option C: Place extracted dataset folder at {CFG.CANDIDATE_DATASET_DIRS[0]}'",
    "    )",
    "    print(error_msg)",
    "    raise FileNotFoundError('Dataset archive or folder missing.')",

    "",
    "# Run dataset resolution",
    "ACTIVE_DATASET_DIR = resolve_and_extract_dataset()"
])

# ==============================================================================
# CELL 6: DATASET VALIDATION
# ==============================================================================
add_markdown_cell([
    "## Step 5: Comprehensive Dataset & Label Integrity Audit",
    "Audits `data.yaml` schema, image/label counts across train/val/test splits, and verifies 100% of label rows follow 5-field YOLO bounding box constraints."
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
    "    # Update data.yaml with absolute paths for Colab environment",
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
    "    print('✅ Dataset & Label Validation 100% PASSED!')",
    "    return data_yaml_path",

    "",
    "DATA_YAML_PATH = audit_dataset_integrity(ACTIVE_DATASET_DIR)"
])

# ==============================================================================
# CELL 7: TRAINING EXECUTION
# ==============================================================================
add_markdown_cell([
    "## Step 6: Final 10-Epoch YOLOv8n GPU Training Run",
    "Train pretrained `yolov8n.pt` for 10 epochs on 100% of the dataset (`fraction=1.0`). Outputs (`best.pt`, `last.pt`, plots, metrics) are saved directly to Google Drive."
])

add_code_cell([
    "import time, os",
    "from ultralytics import YOLO",
    "",
    "print('=' * 75)",
    "print(f'   STARTING FINAL {CFG.EPOCHS}-EPOCH YOLOv8n GPU TRAINING RUN')",
    "print('=' * 75)",
    "print(f'  Data YAML Path   : {DATA_YAML_PATH}')",
    "print(f'  Target Epochs    : {CFG.EPOCHS}')",
    "print(f'  Image Size       : {CFG.IMGSZ}')",
    "print(f'  Batch Size       : {CFG.BATCH_SIZE}')",
    "print(f'  Dataset Fraction : {CFG.FRACTION * 100:.0f}%')",
    "print(f'  Device           : GPU {CFG.DEVICE}')",
    "print(f'  Output Target    : {os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)}')",
    "print('=' * 75)",
    "",
    "# Load baseline pretrained YOLOv8n weights",
    "model = YOLO('yolov8n.pt')",
    "",
    "start_time = time.time()",
    "",
    "# Execute Ultralytics training with direct Drive output persistence",
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
    "    fraction=CFG.FRACTION,",
    "    save=True,",
    "    plots=True,",
    "    verbose=True",
    ")",
    "",
    "elapsed_sec = time.time() - start_time",
    "print('\\n' + '=' * 75)",
    "print(f'✅ GPU TRAINING FINISHED in {elapsed_sec:.2f} seconds ({elapsed_sec/60:.2f} minutes)!')",
    "print('=' * 75)",
    "",
    "# Physical check for best.pt in Google Drive",
    "EXPECTED_BEST_WEIGHTS = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'best.pt')",
    "EXPECTED_LAST_WEIGHTS = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'last.pt')",
    "",
    "if os.path.exists(EXPECTED_BEST_WEIGHTS) and os.path.getsize(EXPECTED_BEST_WEIGHTS) > 0:",
    "    print(f'✅ Physical Weight Check PASSED! best.pt saved at: {EXPECTED_BEST_WEIGHTS} ({os.path.getsize(EXPECTED_BEST_WEIGHTS)} bytes)')",
    "else:",
    "    raise FileNotFoundError(f'❌ Training completed but best.pt was not found at {EXPECTED_BEST_WEIGHTS}')"
])

# ==============================================================================
# CELL 8: CRITICAL CHECKPOINT VERIFICATION
# ==============================================================================
add_markdown_cell([
    "## Step 7: Critical Checkpoint Inspection & Metadata Verification",
    "Safely inspect `best.pt` PyTorch metadata (SHA256 hash, file size, class taxonomy, training epochs, fraction=1.0) to confirm it is the genuine 10-epoch GPU model."
])

add_code_cell([
    "import os, hashlib, torch",
    "",
    "def verify_final_checkpoint():",
    "    best_path = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME, 'weights', 'best.pt')",
    "    print(f'[*] Verifying Newly Trained Checkpoint: {best_path}')",
    "    ",
    "    if not os.path.exists(best_path):",
    "        raise FileNotFoundError(f'Missing checkpoint file at {best_path}')",
    "        ",
    "    size = os.path.getsize(best_path)",
    "    with open(best_path, 'rb') as f:",
    "        sha256 = hashlib.sha256(f.read()).hexdigest()",
    "        ",
    "    print(f'  - File Size : {size} bytes ({size / (1024*1024):.2f} MB)')",
    "    print(f'  - SHA256    : {sha256}')",
    "    ",
    "    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)",
    "    args = ckpt.get('train_args', {})",
    "    if not isinstance(args, dict) and hasattr(args, '__dict__'):",
    "        args = args.__dict__",
    "        ",
    "    train_epochs = args.get('epochs') if isinstance(args, dict) else None",
    "    train_fraction = args.get('fraction') if isinstance(args, dict) else None",
    "    train_batch = args.get('batch') if isinstance(args, dict) else None",
    "    train_device = args.get('device') if isinstance(args, dict) else None",
    "    ",
    "    model_obj = ckpt.get('model', None)",
    "    names = getattr(model_obj, 'names', {})",
    "    if isinstance(names, dict):",
    "        class_list = [names[k] for k in sorted(names.keys())]",
    "    else:",
    "        class_list = list(names)",
    "        ",
    "    print('  - Train Epochs Arg :', train_epochs)",
    "    print('  - Train Fraction   :', train_fraction)",
    "    print('  - Train Batch Size :', train_batch)",
    "    print('  - Device           :', train_device)",
    "    print('  - Class Taxonomy   :', class_list)",
    "    ",
    "    # Verifications",
    "    assert len(class_list) == 5, f'Expected 5 classes, got {len(class_list)}'",
    "    assert set(class_list) == set(CFG.CLASS_NAMES), 'Class names mismatch'",
    "    if train_epochs is not None:",
    "        assert train_epochs == 10, f'Expected 10 epochs, got {train_epochs}'",
    "        ",
    "    print('\\n==================================================')",
    "    print('      [PASS] FINAL MODEL VERIFIED 100% SUCCESS')",
    "    print('==================================================')",
    "    return best_path",

    "",
    "VERIFIED_BEST_PATH = verify_final_checkpoint()"
])

# ==============================================================================
# CELL 9: STEP 6 VALIDATION EVALUATION
# ==============================================================================
add_markdown_cell([
    "## Step 8: Validation Set Performance Evaluation (`val` split)",
    "Load verified `best.pt` from Google Drive and evaluate Precision, Recall, mAP50, and mAP50-95 on the 638 validation images."
])

add_code_cell([
    "import os",
    "from ultralytics import YOLO",
    "",
    "print('[*] Evaluating Validation Performance on verified best.pt...')",
    "best_model = YOLO(VERIFIED_BEST_PATH)",
    "",
    "print('\\n==================================================')",
    "print('    VALIDATION SET PERFORMANCE EVALUATION (val)')",
    "print('==================================================')",
    "val_metrics = best_model.val(data=DATA_YAML_PATH, split='val', device=CFG.DEVICE, verbose=True)",
    "",
    "val_dict = val_metrics.results_dict",
    "val_precision = val_dict.get('metrics/precision(B)', 0.0)",
    "val_recall = val_dict.get('metrics/recall(B)', 0.0)",
    "val_map50 = val_dict.get('metrics/mAP50(B)', 0.0)",
    "val_map50_95 = val_dict.get('metrics/mAP50-95(B)', 0.0)",
    "",
    "print('\\n--- OVERALL VALIDATION METRICS ---')",
    "print(f'  Validation Precision (B) : {val_precision * 100:.2f}% ({val_precision:.4f})')",
    "print(f'  Validation Recall (B)    : {val_recall * 100:.2f}% ({val_recall:.4f})')",
    "print(f'  Validation mAP50 (B)     : {val_map50 * 100:.2f}% ({val_map50:.4f})')",
    "print(f'  Validation mAP50-95 (B)  : {val_map50_95 * 100:.2f}% ({val_map50_95:.4f})')",
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
# CELL 10: STEP 7 TEST EVALUATION
# ==============================================================================
add_markdown_cell([
    "## Step 9: Test Set Generalization Evaluation (`test` split)",
    "Evaluate model generalization metrics on the 319 unseen test set images."
])

add_code_cell([
    "import os",
    "from ultralytics import YOLO",
    "",
    "print('[*] Evaluating Test Set Performance on verified best.pt...')",
    "best_model = YOLO(VERIFIED_BEST_PATH)",
    "",
    "print('\\n==================================================')",
    "print('       TEST SET GENERALIZATION EVALUATION (test)')",
    "print('==================================================')",
    "test_metrics = best_model.val(data=DATA_YAML_PATH, split='test', device=CFG.DEVICE, verbose=True)",
    "",
    "test_dict = test_metrics.results_dict",
    "test_precision = test_dict.get('metrics/precision(B)', 0.0)",
    "test_recall = test_dict.get('metrics/recall(B)', 0.0)",
    "test_map50 = test_dict.get('metrics/mAP50(B)', 0.0)",
    "test_map50_95 = test_dict.get('metrics/mAP50-95(B)', 0.0)",
    "",
    "print('\\n--- OVERALL TEST SET METRICS ---')",
    "print(f'  Test Precision (B) : {test_precision * 100:.2f}% ({test_precision:.4f})')",
    "print(f'  Test Recall (B)    : {test_recall * 100:.2f}% ({test_recall:.4f})')",
    "print(f'  Test mAP50 (B)     : {test_map50 * 100:.2f}% ({test_map50:.4f})')",
    "print(f'  Test mAP50-95 (B)  : {test_map50_95 * 100:.2f}% ({test_map50_95:.4f})')",
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
# CELL 11: STEP 8 INFERENCE BENCHMARK & PREDICTIONS
# ==============================================================================
add_markdown_cell([
    "## Step 10: GPU Inference Latency Benchmark & Visual Predictions",
    "Benchmark GPU inference latency (ms per frame & FPS) and save visual bounding box predictions permanently to Google Drive."
])

add_code_cell([
    "import os, glob, cv2",
    "from ultralytics import YOLO",
    "",
    "best_model = YOLO(VERIFIED_BEST_PATH)",
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
# CELL 12: LOCAL DOWNLOAD PACKAGE (ZIP)
# ==============================================================================
add_markdown_cell([
    "## Step 11: Create Downloadable Artifact Package (`.zip`)",
    "Packages `best.pt`, `last.pt`, `results.csv`, `args.yaml`, training curves, confusion matrix, and visual predictions into `AquaScan_FINAL_10ep_Artifacts.zip` saved to Google Drive and `/content/` for local download."
])

add_code_cell([
    "import os, zipfile",
    "",
    "def create_download_package():",
    "    run_dir = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)",
    "    drive_zip_path = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.FINAL_ZIP_NAME)",
    "    local_zip_path = os.path.join(CFG.COLAB_CONTENT_DIR, CFG.FINAL_ZIP_NAME)",
    "    ",
    "    print(f'[*] Zipping final training artifacts to {drive_zip_path}...')",
    "    ",
    "    with zipfile.ZipFile(drive_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:",
    "        for root, dirs, files in os.walk(run_dir):",
    "            for file in files:",
    "                file_p = os.path.join(root, file)",
    "                arcname = os.path.relpath(file_p, run_dir)",
    "                zipf.write(file_p, arcname)",
    "                ",
    "    # Copy to /content/ for easy Colab file browser download",
    "    import shutil",
    "    shutil.copy2(drive_zip_path, local_zip_path)",
    "    ",
    "    size_mb = os.path.getsize(drive_zip_path) / (1024 * 1024)",
    "    print('\\n==================================================')",
    "    print('       FINAL ARTIFACT ZIP PACKAGE CREATED')",
    "    print('==================================================')",
    "    print(f'  Drive Path : {drive_zip_path}')",
    "    print(f'  Colab Path : {local_zip_path}')",
    "    print(f'  Archive Size: {size_mb:.2f} MB')",
    "    print('==================================================')",
    "    return drive_zip_path",

    "",
    "FINAL_ZIP_PATH = create_download_package()"
])

# ==============================================================================
# CELL 13: FINAL SUMMARY REPORT
# ==============================================================================
add_markdown_cell([
    "## Step 12: Final Pipeline Verification & Summary Report",
    "Prints a clean formatted summary verifying all final deliverables in Google Drive and confirming overall training PASS status."
])

add_code_cell([
    "import os",
    "",
    "run_dir = os.path.join(CFG.DRIVE_PROJECT_DIR, CFG.RUN_NAME)",
    "weights_dir = os.path.join(run_dir, 'weights')",
    "preds_dir = os.path.join(run_dir, 'visual_predictions')",
    "",
    "expected_deliverables = [",
    "    ('best.pt Model Weights', os.path.join(weights_dir, 'best.pt')),",
    "    ('last.pt Model Weights', os.path.join(weights_dir, 'last.pt')),",
    "    ('Results CSV Metrics', os.path.join(run_dir, 'results.csv')),",
    "    ('Training Args YAML', os.path.join(run_dir, 'args.yaml')),",
    "    ('Confusion Matrix Plot', os.path.join(run_dir, 'confusion_matrix.png')),",
    "    ('Training Curves Plot', os.path.join(run_dir, 'results.png')),",
    "    ('Visual Predictions Dir', preds_dir),",
    "    ('Artifact ZIP Package', FINAL_ZIP_PATH)",
    "]",
    "",
    "all_passed = True",
    "print('=' * 80)",
    "print('                 AQUASCAN AI FINAL TRAINING COMPLETE SUMMARY')",
    "print('=' * 80)",
    "print(f'  GPU Accelerator   : {gpu_name if \"gpu_name\" in locals() else \"NVIDIA GPU\"}')",
    "print(f'  Pretrained Model  : yolov8n.pt')",
    "print(f'  Training Epochs   : {CFG.EPOCHS}')",
    "print(f'  Dataset Fraction  : {CFG.FRACTION * 100:.0f}% (Full Dataset)')",
    "print(f'  Classes ({CFG.NUM_CLASSES})   : {\", \".join(CFG.CLASS_NAMES)}')",
    "print(f'  Validation mAP50  : {val_map50 * 100:.2f}% ({val_map50:.4f})' if 'val_map50' in locals() else '')",
    "print(f'  Validation mAP50-95: {val_map50_95 * 100:.2f}% ({val_map50_95:.4f})' if 'val_map50_95' in locals() else '')",
    "print(f'  GPU Throughput    : {fps:.2f} FPS ({avg_total:.2f} ms/frame)' if 'fps' in locals() else '')",
    "print('-' * 80)",
    "print(f'{\"Deliverable Name\":<25} | {\"Status\":<8} | {\"Size (MB/KB)\":<12} | Path')",
    "print('-' * 80)",

    "for label, path in expected_deliverables:",
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
    "        all_passed = False",
    "    print(f'{label:<25} | {status_str:<8} | {size_str:<12} | {path}')",

    "print('=' * 80)",
    "if all_passed:",
    "    print('🚀 FINAL STATUS: [PASS] ALL TRAINING ARTIFACTS PERMANENTLY SAVED TO GOOGLE DRIVE!')",
    "    print(f'   Target Model Path: {os.path.join(weights_dir, \"best.pt\")}')",
    "    print(f'   Download Package : {FINAL_ZIP_PATH}')",
    "else:",
    "    print('⚠️ FINAL STATUS: [FAIL] Some deliverables are missing. Check training log above.')",
    "print('=' * 80)"
])

# Save notebook to workspace root as AquaScan_YOLOv8n_FINAL_GPU_Training.ipynb
out_path = os.path.abspath("AquaScan_YOLOv8n_FINAL_GPU_Training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated new clean final notebook at: {out_path} ({os.path.getsize(out_path)} bytes)")
