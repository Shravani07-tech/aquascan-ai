import os
import time
import base64
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from backend.config import resolve_model_path, CLASS_NAMES, CLASS_COLORS_HEX, CLASS_COLORS_BGR


def _ensure_nms_backend():
    """Keep inference working when torchvision's native extension cannot load.

    Windows Smart App Control can block torchvision\\_C.pyd (WinError 4551), which makes
    `import torchvision` raise "operator torchvision::nms does not exist" inside Ultralytics
    and fails every prediction with HTTP 500. In that case only, expose a minimal
    `torchvision.ops.nms` backed by Ultralytics' own pure-PyTorch NMS (same algorithm).
    No-op when torchvision imports normally.
    """
    import sys
    try:
        import torchvision  # noqa: F401
        return
    except Exception as e:
        print(f"[AquaScan] torchvision native ops unavailable ({e}). "
              f"Using Ultralytics pure-PyTorch NMS fallback.")
    import types
    from ultralytics.utils.nms import TorchNMS
    for name in [m for m in sys.modules if m == "torchvision" or m.startswith("torchvision.")]:
        del sys.modules[name]
    ops = types.ModuleType("torchvision.ops")
    ops.nms = TorchNMS.nms
    tv = types.ModuleType("torchvision")
    tv.ops = ops
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.ops"] = ops


_ensure_nms_backend()

class YOLOInferenceService:
    def __init__(self):
        self.model = None
        self.webcam_model = None
        self.model_path = None
        self.webcam_model_path = None
        self.is_fallback = False
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda:0" if self.cuda_available else "cpu"
        self.active_classes = CLASS_NAMES
        self.reload_model()

    def reload_model(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda:0" if self.cuda_available else "cpu"
        self.model_path = resolve_model_path()
        print(f"Loading YOLO Model [{self.device}] from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            model_name = os.path.basename(self.model_path)
            
            # Inspect loaded model class names to determine trained vs fallback status
            names = getattr(self.model, 'names', {})
            if isinstance(names, dict):
                model_class_list = [names[k] for k in sorted(names.keys())]
            else:
                model_class_list = list(names)

            # Check if model matches the 5 AquaScan classes
            if len(model_class_list) == 5 and set(model_class_list) == set(CLASS_NAMES):
                self.is_fallback = False
                self.active_classes = CLASS_NAMES
                print(f"AquaScan Trained Model loaded: {model_name} (5 classes: {CLASS_NAMES})")
            else:
                self.is_fallback = True
                self.active_classes = model_class_list
                print(f"Fallback Generic Model loaded: {model_name} ({len(model_class_list)} classes)")

        except Exception as e:
            print(f"Error loading YOLO model from {self.model_path}: {e}")
            self.model = None

        # Check for optional webcam-adapted model
        if self.model_path:
            webcam_path = os.path.abspath(os.path.join(os.path.dirname(self.model_path), "best_webcam.pt"))
            if os.path.exists(webcam_path):
                try:
                    self.webcam_model = YOLO(webcam_path)
                    self.webcam_model_path = webcam_path
                    print(f"AquaScan Webcam Model loaded: {os.path.basename(webcam_path)}")
                except Exception as e:
                    print(f"Error loading webcam model from {webcam_path}: {e}")
                    self.webcam_model = None

    def get_status(self):
        active_path = resolve_model_path()
        if active_path != self.model_path:
            self.reload_model()
            
        model_display_name = "AquaScan YOLOv8n" if not self.is_fallback else "Fallback Generic YOLOv8n"
        model_type = "AquaScan Marine Waste Perception" if not self.is_fallback else "Generic COCO Object Detection"
        
        return {
            "success": True,
            "model": model_display_name,
            "model_path": self.model_path,
            "webcam_model_path": self.webcam_model_path,
            "model_type": model_type,
            "classes": self.active_classes,
            "classes_count": len(self.active_classes),
            "device": self.device,
            "cuda_available": self.cuda_available,
            "status": "ready" if self.model is not None else "unavailable",
            "is_fallback": self.is_fallback,
            "colors": CLASS_COLORS_HEX
        }

    def predict(self, image_bytes: bytes, conf_threshold: float = 0.25, hardware_metadata: dict = None, skip_annotated_image: bool = False, capture: dict = None):
        if self.model is None:
            self.reload_model()
            if self.model is None:
                raise RuntimeError("YOLO model is unavailable or failed to initialize.")

        # Decode image bytes safely
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format or corrupt image data.")

        height, width, _ = img.shape

        # Source identifier robust matching
        source_str = str(hardware_metadata.get("source", "")).lower().strip() if hardware_metadata else ""
        is_laptop_webcam = source_str in ["laptop_webcam", "laptop", "webcam", "laptop-webcam", "camera"]

        # Select target model (use webcam model if available and request source is laptop_webcam)
        target_model = self.model
        if is_laptop_webcam and self.webcam_model is not None:
            target_model = self.webcam_model

        # Force imgsz=416 for laptop webcam requests to match feature scale; imgsz=640 for marine sources
        img_size = 416 if is_laptop_webcam else 640

        # Debug frame collector (AQUASCAN_SAVE_FRAMES=1): record exactly what the model receives
        if capture is not None:
            capture.update(img=img, imgsz=img_size, conf=conf_threshold, model=target_model, device=self.device)

        # Run inference
        t0 = time.time()
        results = target_model.predict(source=img, conf=conf_threshold, imgsz=img_size, device=self.device, verbose=False)
        t1 = time.time()

        detections = []
        category_counts = {name: 0 for name in CLASS_NAMES}

        # Annotated image canvas (only created if needed)
        annotated_img = img.copy() if not skip_annotated_image else None

        preprocess_ms = 0.0
        inference_ms = (t1 - t0) * 1000.0
        postprocess_ms = 0.0

        for r in results:
            if hasattr(r, 'speed') and isinstance(r.speed, dict):
                preprocess_ms = r.speed.get('preprocess', 0.0)
                inference_ms = r.speed.get('inference', inference_ms)
                postprocess_ms = r.speed.get('postprocess', 0.0)

            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                # Validate class index boundaries
                if 0 <= cls_id < len(self.active_classes):
                    cls_name = self.active_classes[cls_id]
                else:
                    cls_name = f"unknown_{cls_id}"

                if cls_name in category_counts:
                    category_counts[cls_name] += 1

                x1_norm = round(x1 / width, 4)
                y1_norm = round(y1 / height, 4)
                x2_norm = round(x2 / width, 4)
                y2_norm = round(y2 / height, 4)

                color_hex = CLASS_COLORS_HEX.get(cls_name, "#ffffff")

                detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_normalized": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "color": color_hex
                })

                if annotated_img is not None:
                    color_bgr = CLASS_COLORS_BGR.get(cls_name, (255, 255, 255))
                    cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)
                    label_text = f"{cls_name.upper()} {confidence*100:.1f}%"
                    cv2.putText(annotated_img, label_text, (int(x1), max(int(y1) - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        if is_laptop_webcam:
            frame_id_val = hardware_metadata.get("frame_id") if hardware_metadata else "N/A"
            model_file_val = os.path.basename(self.model_path) if self.model_path else "best.pt"
            print("\n" + "="*60)
            print("[AquaScan Live Trace]")
            print(f"  SOURCE RECEIVED     : {source_str}")
            print(f"  CONF RECEIVED       : {conf_threshold}")
            print(f"  FRAME ID            : {frame_id_val}")
            print(f"  IMAGE SIZE          : {width} x {height}")
            print(f"  MODEL FILE          : {model_file_val}")
            print(f"  IMGSZ               : {img_size}")
            print(f"  YOLO DETECTION COUNT: {len(detections)}")
            print(f"  YOLO DETECTIONS     : {detections}")
            print(f"  API RESPONSE COUNT  : {len(detections)}")
            print("="*60 + "\n")

        total_ms = preprocess_ms + inference_ms + postprocess_ms
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        # Base64 annotated image URL (only generated when required)
        annotated_data_url = None
        if not skip_annotated_image and annotated_img is not None:
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_str = base64.b64encode(buffer).decode('utf-8')
            annotated_data_url = f"data:image/jpeg;base64,{base64_str}"

        response_data = {
            "success": True,
            "model": "AquaScan YOLOv8n" if not self.is_fallback else "Fallback Generic YOLOv8n",
            "model_path": self.model_path,
            "is_fallback": self.is_fallback,
            "count": len(detections),
            "category_counts": category_counts,
            "detections": detections,
            "latency_ms": {
                "preprocess": round(preprocess_ms, 2),
                "inference": round(inference_ms, 2),
                "postprocess": round(postprocess_ms, 2),
                "total": round(total_ms, 2),
                "fps": round(fps, 2)
            },
            "image": annotated_data_url,
            "image_dimensions": {"width": width, "height": height}
        }

        if hardware_metadata:
            response_data["hardware_metadata"] = hardware_metadata

        return response_data

# Global singleton instance
inference_service = YOLOInferenceService()

