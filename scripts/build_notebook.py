import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🛸 AquaScan AI — Full 30-Epoch YOLOv8n GPU Training Notebook\n",
                    "\n",
                    "This Jupyter Notebook executes the complete **30-epoch baseline training experiment** for **AquaScan AI** using a **GPU (T4)** environment.\n",
                    "\n",
                    "### 📋 Notebook Pipeline Workflow:\n",
                    "1. **GPU Verification:** Check CUDA & NVIDIA T4 accelerator status.\n",
                    "2. **Environment Setup:** Install `ultralytics`, `opencv-python-headless`, `pyyaml`, and `matplotlib`.\n",
                    "3. **Package Extraction:** Extract `AquaScan_GPU_Training_Pack.zip` into Colab workspace.\n",
                    "4. **Dataset & YAML Verification:** Validate `data.yaml` and verify 100% of labels are 5-field YOLO bounding boxes.\n",
                    "5. **30-Epoch Model Training:** Train pretrained `yolov8n.pt` (`imgsz=640`, `batch=16`, `device=0`).\n",
                    "6. **Validation Evaluation:** Evaluate Precision, Recall, mAP50, mAP50-95, and per-class metrics on 638 validation images.\n",
                    "7. **Test Set Evaluation:** Evaluate generalization performance on 319 test images.\n",
                    "8. **Inference Latency Benchmark:** Benchmark GPU inference speed & generate visual test predictions.\n",
                    "9. **Results Packaging:** Package `best.pt`, `last.pt`, `results.csv`, confusion matrix, curves, and predictions into `AquaScan_YOLOv8n_Full_30ep_Results.zip` for instant download.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Verify NVIDIA GPU & CUDA Acceleration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!nvidia-smi\n",
                    "import torch\n",
                    "print('CUDA Available:', torch.cuda.is_available())\n",
                    "if torch.cuda.is_available():\n",
                    "    print('GPU Device Name:', torch.cuda.get_device_name(0))\n",
                    "else:\n",
                    "    print('WARNING: GPU not detected. Go to Runtime > Change runtime type > Select GPU (T4).')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Install Ultralytics and Dependencies"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q ultralytics opencv-python-headless pyyaml matplotlib pandas"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Extract GPU Training Package"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, zipfile\n",
                    "\n",
                    "zip_path = '/content/AquaScan_GPU_Training_Pack.zip'\n",
                    "if os.path.exists(zip_path):\n",
                    "    print(f'Extracting {zip_path}...')\n",
                    "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
                    "        zip_ref.extractall('/content/')\n",
                    "    print('Extraction complete!')\n",
                    "else:\n",
                    "    print(f'ERROR: {zip_path} not found. Please upload AquaScan_GPU_Training_Pack.zip to /content/')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Configure data.yaml & Verify Label Integrity"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, yaml\n",
                    "\n",
                    "clean_dir = '/content/dataset_clean/floating_waste_v10'\n",
                    "data_yaml_path = os.path.join(clean_dir, 'data.yaml')\n",
                    "\n",
                    "yaml_content = {\n",
                    "    'path': clean_dir,\n",
                    "    'train': 'train/images',\n",
                    "    'val': 'valid/images',\n",
                    "    'test': 'test/images',\n",
                    "    'nc': 5,\n",
                    "    'names': ['bottle', 'can', 'carton', 'paper', 'plastic']\n",
                    "}\n",
                    "os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)\n",
                    "with open(data_yaml_path, 'w') as f:\n",
                    "    yaml.dump(yaml_content, f)\n",
                    "\n",
                    "print('Updated data.yaml path:', data_yaml_path)\n",
                    "\n",
                    "# Label Verification\n",
                    "splits = ['train', 'valid', 'test']\n",
                    "total_labels, total_boxes, invalid_boxes = 0, 0, 0\n",
                    "for split in splits:\n",
                    "    lbl_dir = os.path.join(clean_dir, split, 'labels')\n",
                    "    for fname in os.listdir(lbl_dir):\n",
                    "        total_labels += 1\n",
                    "        with open(os.path.join(lbl_dir, fname), 'r') as f:\n",
                    "            lines = f.read().strip().split('\\n')\n",
                    "        for line in lines:\n",
                    "            parts = line.strip().split()\n",
                    "            if not parts: continue\n",
                    "            total_boxes += 1\n",
                    "            if len(parts) != 5:\n",
                    "                invalid_boxes += 1\n",
                    "\n",
                    "print(f'Verified {total_labels} label files and {total_boxes} bounding boxes.')\n",
                    "print(f'Invalid Bounding Boxes: {invalid_boxes}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Train YOLOv8n on GPU (30 Epochs)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from ultralytics import YOLO\n",
                    "import time\n",
                    "\n",
                    "weights_path = '/content/yolov8n.pt' if os.path.exists('/content/yolov8n.pt') else 'yolov8n.pt'\n",
                    "model = YOLO(weights_path)\n",
                    "\n",
                    "project_dir = '/content/outputs/training'\n",
                    "run_name = 'yolov8n_full_30ep'\n",
                    "\n",
                    "print('Starting 30-Epoch GPU Training...')\n",
                    "start_time = time.time()\n",
                    "\n",
                    "model.train(\n",
                    "    data=data_yaml_path,\n",
                    "    epochs=30,\n",
                    "    imgsz=640,\n",
                    "    batch=16,\n",
                    "    workers=4,\n",
                    "    device=0,\n",
                    "    project=project_dir,\n",
                    "    name=run_name,\n",
                    "    exist_ok=True,\n",
                    "    fraction=1.0,\n",
                    "    save=True,\n",
                    "    plots=True\n",
                    ")\n",
                    "\n",
                    "elapsed = time.time() - start_time\n",
                    "print(f'GPU Training finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Evaluate Validation and Test Performance"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "run_dir = os.path.join(project_dir, run_name)\n",
                    "best_weights = os.path.join(run_dir, 'weights', 'best.pt')\n",
                    "best_model = YOLO(best_weights)\n",
                    "\n",
                    "print('--- VALIDATION SET EVALUATION (638 images) ---')\n",
                    "val_results = best_model.val(data=data_yaml_path, split='val')\n",
                    "print('Validation Precision:', val_results.results_dict.get('metrics/precision(B)'))\n",
                    "print('Validation Recall:', val_results.results_dict.get('metrics/recall(B)'))\n",
                    "print('Validation mAP50:', val_results.results_dict.get('metrics/mAP50(B)'))\n",
                    "print('Validation mAP50-95:', val_results.results_dict.get('metrics/mAP50-95(B)'))\n",
                    "\n",
                    "print('\\n--- TEST SET EVALUATION (319 images) ---')\n",
                    "test_results = best_model.val(data=data_yaml_path, split='test')\n",
                    "print('Test Precision:', test_results.results_dict.get('metrics/precision(B)'))\n",
                    "print('Test Recall:', test_results.results_dict.get('metrics/recall(B)'))\n",
                    "print('Test mAP50:', test_results.results_dict.get('metrics/mAP50(B)'))\n",
                    "print('Test mAP50-95:', test_results.results_dict.get('metrics/mAP50-95(B)'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 7: Run Test Predictions & Benchmark Inference Speed"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import cv2\n",
                    "\n",
                    "predictions_dir = os.path.join(run_dir, 'predictions')\n",
                    "os.makedirs(predictions_dir, exist_ok=True)\n",
                    "\n",
                    "test_img_dir = os.path.join(clean_dir, 'test', 'images')\n",
                    "test_images = sorted(os.listdir(test_img_dir))[:10]\n",
                    "\n",
                    "speed_pre, speed_inf, speed_post = [], [], []\n",
                    "\n",
                    "for img_name in test_images:\n",
                    "    img_path = os.path.join(test_img_dir, img_name)\n",
                    "    res = best_model.predict(source=img_path, device=0, verbose=False)\n",
                    "    for r in res:\n",
                    "        speed_pre.append(r.speed.get('preprocess', 0.0))\n",
                    "        speed_inf.append(r.speed.get('inference', 0.0))\n",
                    "        speed_post.append(r.speed.get('postprocess', 0.0))\n",
                    "        annotated_img = r.plot()\n",
                    "        cv2.imwrite(os.path.join(predictions_dir, img_name), annotated_img)\n",
                    "\n",
                    "avg_pre = sum(speed_pre) / len(speed_pre)\n",
                    "avg_inf = sum(speed_inf) / len(speed_inf)\n",
                    "avg_post = sum(speed_post) / len(speed_post)\n",
                    "avg_tot = avg_pre + avg_inf + avg_post\n",
                    "fps = 1000.0 / avg_tot if avg_tot > 0 else 0.0\n",
                    "\n",
                    "print(f'Average GPU Preprocess Latency: {avg_pre:.2f} ms')\n",
                    "print(f'Average GPU Inference Latency: {avg_inf:.2f} ms')\n",
                    "print(f'Average GPU Postprocess Latency: {avg_post:.2f} ms')\n",
                    "print(f'Average Total Frame Latency: {avg_tot:.2f} ms ({fps:.2f} FPS)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 8: Package Complete Results for Download"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "output_zip = '/content/AquaScan_YOLOv8n_Full_30ep_Results.zip'\n",
                    "!zip -r {output_zip} /content/outputs/training/yolov8n_full_30ep\n",
                    "print('\\n' + '='*60)\n",
                    "print(f'RESULTS ZIPPED SUCCESSFULLY TO: {output_zip}')\n",
                    "print('Download this file and extract into your local workspace at:')\n",
                    "print('outputs/training/yolov8n_full_30ep/')\n",
                    "print('='*60)"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    target_path = "AquaScan_YOLOv8n_GPU_Training.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Created notebook: {os.path.abspath(target_path)}")

if __name__ == "__main__":
    create_notebook()
