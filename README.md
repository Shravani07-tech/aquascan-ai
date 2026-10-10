# 🛸 AquaScan AI — Marine Waste Perception Platform

**System Status:** `SOFTWARE FREEZE COMPLETE — PRODUCTION & DEMO READY`  
**Physical Hardware Status:** `NOT TESTED (Raspberry Pi and Pi Camera hardware are at college connected to the network)`

AquaScan AI is an edge-ready computer vision perception system engineered to detect floating marine waste (bottles, cans, cartons, paper, plastics) from camera footage. The software platform includes a custom-trained YOLOv8 object detection model, a high-performance FastAPI REST backend, a marine dark glassmorphism dashboard, and a software-only edge hardware simulator.

---

## 🏗️ System Architecture & Data Flow

```text
  ┌─────────────────────────────────────────────────────────────┐
  │              Raspberry Pi Edge Client Node                  │
  │     - Pi Camera Sensor (JPEG Frame Capture)                 │
  │     - hardware/rpi_client.py (HTTP POST Client)              │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 │ HTTP POST /api/predict
                                 │ (JPEG Frame + Hardware Metadata)
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 PC AquaScan Backend Server                  │
  │     - FastAPI REST API Service                              │
  │     - PyTorch / Ultralytics YOLOv8 Inference Engine         │
  │     - OpenCV High-Contrast Bounding Box Annotator           │
  └──────────────┬──────────────────────────────┬───────────────┘
                 │                              │
                 ▼                              ▼
  ┌──────────────────────────────┐ ┌─────────────────────────────┐
  │    AquaScan YOLOv8 Model     │ │     Web Frontend UI         │
  │  models/best.pt (5 classes)  │ │ (HTML5 Canvas Dashboard)    │
  └──────────────────────────────┘ └─────────────────────────────┘
```

### Complete End-to-End Data Pipeline:
```text
Pi Camera Sensor
  └──► Raspberry Pi Capture
        └──► JPEG Frame + Metadata (source, device_id, camera_id, frame_id)
              └──► HTTP POST /api/predict over LAN
                    └──► PC FastAPI Server (0.0.0.0:8000)
                          └──► YOLOv8n Trained Perception Model (models/best.pt)
                                └──► Structured JSON Detections + Base64 Overlay
                                      └──► Raspberry Pi Edge Client Response
                                            └──► Live Web Frontend Dashboard
```

---

## 📊 Model Specifications & Training Metrics

The active production model (`models/best.pt`) was trained in Google Colab on the Floating Waste v10 dataset (6,687 training images, 638 validation images):

| Metric | Specification | Details |
|---|---|---|
| **Classes (5)** | `bottle`, `can`, `carton`, `paper`, `plastic` | Marine floating waste taxonomy |
| **Validation Precision** | **79.4%** (0.794) | Precision accuracy on validation set |
| **Validation Recall** | **65.0%** (0.650) | Sensitivity across floating waste items |
| **mAP50** | **69.4%** (0.694) | Mean Average Precision at IoU 0.50 |
| **mAP50-95** | **40.0%** (0.400) | Strict IoU threshold mAP |
| **Training Hardware** | YOLOv8n (10 Epochs, Tesla T4 GPU) | High-throughput nano detection backbone |
| **Active Model Path** | `models/best.pt` (6.22 MB) | Verified 5-class AquaScan model |
| **Fallback Mode** | `is_fallback: false` | Trained model active (*generic yolov8n.pt NOT used*) |

### Dataset source (not in git — re-download it)

Datasets and model weights are **not** in this repository (see `.gitignore`).

- **Dataset:** *Floating Waste* **v10** on Roboflow Universe, workspace `garbagedetection-gqqya`, project
  `floating-waste`: https://universe.roboflow.com/garbagedetection-gqqya/floating-waste/dataset/10
- **License:** CC BY 4.0 (attribute the Roboflow Universe project above). Exported 2025-01-21 in **YOLOv8** format,
  7,644 images (train 6,687 / valid 638 / test 319), 5 classes `bottle, can, carton, paper, plastic`; Roboflow
  pre-processing: auto-orient, resize to fit 640×640; augmentation ×3: rotation ±15°, exposure ±15%.
- **Re-download:** open the link above → *Download Dataset* → format **YOLOv8** → unzip to
  `dataset_raw/floating_waste_v10/`. Then run `.venv\Scripts\python.exe scripts\clean_dataset.py` to rebuild
  `dataset_clean/floating_waste_v10/` (converts 27 polygon annotations to boxes), and
  `.venv\Scripts\python.exe scripts\verify_clean_dataset.py` (expect 7,644 label files, 21,111 boxes, 100% valid).
- **Model weights:** `models/best.pt` (SHA-256 `d8248744d7d5ae2758f61637572f6726f60e938ceb53191be47ac66195741b78`) is
  backed up outside git (OneDrive `AquaScan-backup\best.pt`). Verify the hash after restoring.

---

## 📁 Repository Structure

```text
AquaScan-AI/
├── backend/                      # FastAPI Backend REST Service
│   ├── config.py                 # Host/Port, model path resolver & color themes
│   ├── main.py                   # REST endpoints, input validators & static UI router
│   └── services/
│       ├── inference.py          # Ultralytics YOLO inference, status inspector & annotator
│       └── samples.py            # Demo sample gallery provider & thumbnail generator
├── frontend/                     # Marine Glassmorphism Web Application
│   ├── index.html                # Main workspace layout, dropzone, canvas viewer & stat panels
│   ├── css/style.css             # Dark marine theme, glassmorphism tokens & animations
│   └── js/app.js                 # Drag-and-drop controller, HTML5 canvas renderer & API client
├── hardware/                     # Edge Client Reference Architecture
│   └── rpi_client.py             # Standalone Raspberry Pi reference client script
├── models/                       # Active Production Model Directory
│   └── best.pt                   # Verified 5-class AquaScan trained model weights
├── outputs/                      # Training reports, confusion matrices & metrics
├── scripts/                      # Automated test suite & auditing scripts
│   ├── test_api.py               # Complete 13-point API integration test suite
│   ├── test_real_detection.py    # Honest multi-threshold test set evaluation script
│   └── test_hardware_simulation.py # Software-only Raspberry Pi edge simulation script
└── run_server.py                 # Application server launcher
```

---

## ⚡ Model Path Resolution Hierarchy

`backend/config.py` resolves weights dynamically in the following order:

1. `MODEL_PATH` Environment Variable
2. `models/best.pt` (**Active Production Model**)
3. `outputs/training/yolov8n_full_10ep/weights/best.pt`
4. `outputs/training/yolov8n_baseline/weights/best.pt`
5. `yolov8n.pt` (*Generic COCO 80-class fallback only*)

The system inspects model class names dynamically and reports `is_fallback: false` when 5 AquaScan classes are loaded.

---

## 🌐 API Endpoint Specifications

### 1. `GET /api/status`
Returns active model status, device configuration (`cuda`/`cpu`), loaded classes, and fallback status.

### 2. `GET /api/samples`
Returns curated demo sample metadata including valid web image URLs (`image_url`).

### 3. `GET /api/samples/{sample_id}/image`
Serves raw JPEG bytes for demo sample thumbnails and previews.

### 4. `POST /api/predict`
Main prediction endpoint. Accepts an uploaded image file (`file`) OR demo sample ID (`sample_id`), `conf_threshold`, and optional hardware metadata (`source`, `device_id`, `camera_id`, `frame_id`, `timestamp`).

#### Example Request:
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@test.jpg" \
  -F "conf_threshold=0.25" \
  -F "source=raspberry_pi" \
  -F "device_id=aquascan-pi-01"
```

---

## 💻 Software Startup Guide

### 1. Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Launch the Application Server
```bash
python run_server.py
```
Server binds to `0.0.0.0:8000` (configurable via `AQUASCAN_HOST` and `AQUASCAN_PORT`).

### 3. Access Web UI & Docs
- **Dashboard UI:** [http://localhost:8000](http://localhost:8000)
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Automated Testing & Simulation

Run the test suite commands:

```bash
# 1. Honest multi-threshold test set evaluation
python scripts/test_real_detection.py

# 2. Complete 13-point API integration test suite
python scripts/test_api.py

# 3. Software-only Raspberry Pi edge simulation test
python scripts/test_hardware_simulation.py
```

---

## 📡 Network Configuration for Raspberry Pi (LAN)

When the Raspberry Pi is deployed on the same Wi-Fi/LAN:

### 1. Find PC LAN IP Address (on Windows):
```cmd
ipconfig
```
Look for `IPv4 Address` under Wireless LAN or Ethernet (e.g., `192.168.1.100`).

### 2. Configure Raspberry Pi Server Target:
Set the environment variable or `--server` argument on the Pi:
```bash
export AQUASCAN_SERVER_URL="http://192.168.1.100:8000"
```

### 3. Test Network Reachability from Raspberry Pi:
```bash
# Check ping
ping 192.168.1.100

# Check status endpoint
curl http://192.168.1.100:8000/api/status
```

---

## 🍓 Future Physical Hardware Integration

*(To be performed when physical Raspberry Pi and Pi Camera are available)*

### Step 1: Install Dependencies on Raspberry Pi
```bash
pip install requests picamera2  # or opencv-python
```

### Step 2: Run Edge Client Script on Raspberry Pi
```bash
python hardware/rpi_client.py --server http://<PC_LAN_IP>:8000 --device-id aquascan-pi-college --conf 0.10
```

---

## 🔧 Troubleshooting

### Port Conflict (`WinError 10048`):
If port 8000 is occupied:
```cmd
cmd > netstat -ano | findstr :8000
cmd > taskkill /PID <PID> /F
```
Or start server on a custom port:
```cmd
cmd > set AQUASCAN_PORT=8001 && python run_server.py
```
