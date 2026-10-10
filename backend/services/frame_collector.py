"""Debug-only frame collector (handoff v2 §14).

Enabled only when the backend starts with AQUASCAN_SAVE_FRAMES=1; otherwise every call is a
no-op and request handling is unchanged.

For each /api/predict request it writes, under debug_frames/<phase>/:
  <seq>_<timestamp>.jpg   the exact image bytes the server received. cv2.imdecode(..., IMREAD_COLOR)
                          of these bytes reproduces the array passed to YOLO bit-for-bit.
  <seq>_<timestamp>.json  path (A/B/upload), source, client frame sequence, effective imgsz and
                          confidence (after any caps), shape, dtype, the deployed detections, and
                          all raw boxes down to conf 0.01 from a separate debug inference at the
                          same imgsz.

Phase name, first match wins: ?phase= on /api/predict, ?phase= in the page URL (sent as the
Referer by the browser, so http://localhost:8000/?phase=p2 tags Path A frames), the
AQUASCAN_PHASE env var (used for Path B, whose client is not modified), else "unsorted".
"""
import hashlib
import json
import os
import re
import threading
import time
from urllib.parse import parse_qs, urlparse

from backend.config import BASE_DIR

ENABLED = os.getenv("AQUASCAN_SAVE_FRAMES") == "1"
OUTPUT_DIR = os.path.join(BASE_DIR, "debug_frames")
RAW_CONF_FLOOR = 0.01

_lock = threading.Lock()
_next_seq = {}

if ENABLED:
    print("=" * 65)
    print(f"  FRAME COLLECTOR ON -> {OUTPUT_DIR}")
    print(f"  AQUASCAN_PHASE = {os.getenv('AQUASCAN_PHASE') or '(not set: frames go to ?phase= or unsorted)'}")
    print("  Debug only: every frame runs inference twice. Do NOT present with this on.")
    print("=" * 65)


def _clean(name):
    name = re.sub(r"[^A-Za-z0-9_-]", "", name or "")[:40]
    return name or None


def resolve_phase(request):
    phase = _clean(request.query_params.get("phase"))
    if not phase:
        referer = request.headers.get("referer", "")
        phase = _clean(parse_qs(urlparse(referer).query).get("phase", [""])[0])
    return phase or _clean(os.getenv("AQUASCAN_PHASE")) or "unsorted"


def _path_label(source):
    s = (source or "").lower().strip()
    if s in ("laptop_webcam", "laptop", "webcam", "laptop-webcam", "camera"):
        return "A"
    if s == "raspberry_pi":
        return "B"
    return "upload"


def _reserve_seq(phase_dir):
    with _lock:
        if phase_dir not in _next_seq:
            os.makedirs(phase_dir, exist_ok=True)
            existing = [int(f.split("_")[0]) for f in os.listdir(phase_dir)
                        if f.endswith(".json") and f.split("_")[0].isdigit()]
            _next_seq[phase_dir] = max(existing, default=0) + 1
        seq = _next_seq[phase_dir]
        _next_seq[phase_dir] += 1
        return seq


def _boxes(results, names):
    out = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            out.append({
                "class": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id],
                "class_id": cls_id,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
    return sorted(out, key=lambda d: -d["confidence"])


def save(phase, image_bytes, capture, source, frame_id, conf_requested, response):
    """Write the frame and its metadata. Never raises: a collector failure must not break inference."""
    try:
        if not capture or "img" not in capture:
            return
        phase_dir = os.path.join(OUTPUT_DIR, phase)
        seq = _reserve_seq(phase_dir)
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"{int(now % 1 * 1000):03d}"
        base = os.path.join(phase_dir, f"{seq:05d}_{stamp}")

        img, model = capture["img"], capture["model"]
        raw = model.predict(source=img, conf=RAW_CONF_FLOOR, imgsz=capture["imgsz"],
                            device=capture["device"], verbose=False)

        with open(base + ".jpg", "wb") as f:
            f.write(image_bytes)
        meta = {
            "phase": phase,
            "seq": seq,
            "server_received_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}",
            "path": _path_label(source),
            "source": source,
            "client_frame_seq": frame_id,
            "image_file": os.path.basename(base + ".jpg"),
            "image_bytes": len(image_bytes),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "decode": "cv2.imdecode(bytes, cv2.IMREAD_COLOR) -> BGR; identical to the model input array",
            "shape": list(img.shape),
            "dtype": str(img.dtype),
            "imgsz": capture["imgsz"],
            "conf_requested": conf_requested,
            "conf_effective": capture["conf"],
            "model_path": response.get("model_path"),
            "deployed_detections": [
                {k: d[k] for k in ("class", "class_id", "confidence", "bbox")}
                for d in response.get("detections", [])
            ],
            "raw_conf_floor": RAW_CONF_FLOOR,
            "raw_boxes": _boxes(raw, model.names),
        }
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    except Exception as e:
        print(f"[AquaScan FrameCollector] failed to save frame: {e}")
