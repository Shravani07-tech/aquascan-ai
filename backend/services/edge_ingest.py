"""
AquaScan AI — Edge-inference ingest (Pi detects, dashboard displays).

The Raspberry Pi runs best.pt itself (hardware/pi_edge_infer.py --server) and POSTs the frame plus its
detections to /api/edge/ingest. No inference happens on the laptop for this path: the payload is validated
and converted into the same prediction format /api/predict produces, so the existing Raspberry Pi dashboard
panel can draw it unchanged.
"""
import json
import time

import cv2
import numpy as np

from backend.config import CLASS_NAMES, CLASS_COLORS_HEX


class EdgeIngestError(ValueError):
    """Invalid edge payload (reported to the client as HTTP 400)."""


def _num(meta, key, default=None):
    v = meta.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EdgeIngestError(f"'{key}' must be a number")
    return float(v)


def parse_meta(meta_str: str) -> dict:
    try:
        meta = json.loads(meta_str)
    except (TypeError, json.JSONDecodeError) as e:
        raise EdgeIngestError(f"'meta' is not valid JSON: {e}")
    if not isinstance(meta, dict):
        raise EdgeIngestError("'meta' must be a JSON object")
    dets = meta.get("detections", [])
    if not isinstance(dets, list):
        raise EdgeIngestError("'detections' must be a list")
    for i, d in enumerate(dets):
        if not isinstance(d, dict):
            raise EdgeIngestError(f"detection {i} must be an object")
        if d.get("class") not in CLASS_NAMES:
            raise EdgeIngestError(f"detection {i}: unknown class {d.get('class')!r}")
        c = d.get("confidence")
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
            raise EdgeIngestError(f"detection {i}: confidence must be a number in [0, 1]")
        b = d.get("bbox")
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)):
            raise EdgeIngestError(f"detection {i}: bbox must be [x1, y1, x2, y2]")
    return meta


def build_state(image_bytes: bytes, meta: dict):
    """Validate the frame and return (prediction, hardware_metadata) for rpi_stream_store."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise EdgeIngestError("Invalid image format or corrupt image data.")
    height, width = img.shape[:2]

    detections, counts = [], {name: 0 for name in CLASS_NAMES}
    for d in meta.get("detections", []):
        x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
        counts[d["class"]] += 1
        detections.append({
            "class": d["class"],
            "class_id": CLASS_NAMES.index(d["class"]),
            "confidence": round(float(d["confidence"]), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_normalized": [round(x1 / width, 4), round(y1 / height, 4), round(x2 / width, 4), round(y2 / height, 4)],
            "color": CLASS_COLORS_HEX.get(d["class"], "#ffffff"),
        })

    inference_ms = _num(meta, "inference_ms")
    fps = _num(meta, "fps")
    received_at = time.time()
    prediction = {
        "success": True,
        "model": "AquaScan YOLOv8n (edge)",
        "count": len(detections),
        "category_counts": counts,
        "detections": detections,
        "latency_ms": {"total": round(inference_ms or 0.0, 1), "fps": round(fps or 0.0, 2)},
        "image": None,
        "image_dimensions": {"width": width, "height": height},
    }
    metadata = {
        "source": "raspberry_pi_edge",
        "inference_location": "edge",
        "device_id": str(meta.get("device_id") or "aquascan-pi"),
        "camera_id": str(meta.get("camera_id") or "picamera"),
        "frame_id": meta.get("frame_id"),
        "imgsz": meta.get("imgsz"),
        "conf": meta.get("conf"),
        "inference_ms": inference_ms,
        "edge_fps": fps,
        "cpu_temp_c": _num(meta, "cpu_temp_c"),
        "captured_at": _num(meta, "captured_at"),
        "received_at": received_at,
    }
    return prediction, metadata
