import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Server configuration for LAN / Raspberry Pi readiness
HOST = os.getenv("AQUASCAN_HOST", "0.0.0.0")
PORT = int(os.getenv("AQUASCAN_PORT", "8000"))

# Model path resolution priority:
# 1. MODEL_PATH environment variable
# 2. models/best.pt
# 3. outputs/training/yolov8n_full_10ep/weights/best.pt
# 4. outputs/training/yolov8n_baseline/weights/best.pt
# 5. yolov8n.pt (generic fallback)
DEFAULT_MODEL_CANDIDATES = [
    os.getenv("MODEL_PATH"),
    os.path.join(BASE_DIR, "models", "best.pt"),
    os.path.join(BASE_DIR, "outputs", "training", "yolov8n_full_10ep", "weights", "best.pt"),
    os.path.join(BASE_DIR, "outputs", "training", "yolov8n_baseline", "weights", "best.pt"),
    os.path.join(BASE_DIR, "yolov8n.pt"),
    "yolov8n.pt"
]

def resolve_model_path():
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate and os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return os.path.abspath(candidate)
    return "yolov8n.pt"

# Class Names
CLASS_NAMES = ['bottle', 'can', 'carton', 'paper', 'plastic']

# High-contrast marine theme colors (Hex for frontend, BGR for OpenCV rendering)
CLASS_COLORS_HEX = {
    'bottle': '#00f5d4',   # Neon Aqua
    'can': '#00bbf9',      # Electric Cyan
    'carton': '#f15bb5',   # Vivid Pink / Magenta
    'paper': '#fee440',    # Solar Yellow
    'plastic': '#ff0054'   # Neon Red / Coral
}

CLASS_COLORS_BGR = {
    'bottle': (212, 245, 0),
    'can': (249, 187, 0),
    'carton': (181, 91, 241),
    'paper': (64, 228, 254),
    'plastic': (84, 0, 255)
}

# Upload Constraints
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

