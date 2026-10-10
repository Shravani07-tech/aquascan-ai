# 🤖 AquaScan AI Model Weights Directory

Place your final trained PyTorch model weights (`best.pt`) in this directory:

```text
AquaScan-AI/
└── models/
    └── best.pt  <-- Place your trained weights here
```

## How It Works

The AquaScan AI backend (`backend/config.py`) automatically detects `best.pt` in the following priority order:

1. `MODEL_PATH` Environment Variable
2. `models/best.pt` (**Recommended**)
3. `outputs/training/yolov8n_full_30ep/weights/best.pt`
4. `outputs/training/yolov8n_baseline/weights/best.pt`
5. `yolov8n.pt` (Fallback pretrained model)

Simply copying `best.pt` into `models/best.pt` instantly updates the application without requiring any code changes or server restarts!
