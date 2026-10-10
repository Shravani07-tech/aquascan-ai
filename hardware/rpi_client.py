#!/usr/bin/env python3
"""
AquaScan AI — Raspberry Pi Edge Camera Transmission Client

Features:
- Primary capture backend: Picamera2 (Raspberry Pi OS Bookworm libcamera)
- Secondary capture backend: OpenCV cv2.VideoCapture(0) (Webcam / USB camera)
- Tertiary fallback: Local test image generator / sequence reader
- In-memory JPEG encoding (zero disk I/O latency)
- Continuous frame transmission with configurable FPS / interval
- Auto-reconnect & connection retry handling
- Hardware metadata format fully compatible with AquaScan FastAPI /api/predict

Usage Examples:
  1. Default Pi Camera / OpenCV Loop Mode (2-3 FPS):
     python hardware/rpi_client.py --server http://192.168.1.100:8000 --loop --interval 0.3

  2. Local Test Image Stream Mode (offline testing):
     python hardware/rpi_client.py --server http://localhost:8000 --image test.jpg --loop --interval 0.5
"""

import os
import sys
import time
import argparse
import logging
import io

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' package is required. Install via: pip install requests")
    sys.exit(1)

# Try importing OpenCV for fallback capture & JPEG encoding
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Try importing Picamera2 for Raspberry Pi native libcamera access
try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except ImportError:
    PICAM2_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AquaScan-PiClient")


class CameraBackend:
    """Camera abstraction supporting Picamera2, OpenCV, and Synthetic/File Fallbacks."""

    def __init__(self, mode="auto", width=640, height=480, fallback_image=None, rotate=0):
        self.mode = mode
        self.width = width
        self.height = height
        self.fallback_image = fallback_image
        self.rotate = rotate
        self.active_backend = None
        self.cam_obj = None
        
        self._initialize_camera()

    def _initialize_camera(self):
        if self.mode in ("auto", "picamera2") and PICAM2_AVAILABLE:
            try:
                logger.info("Initializing Raspberry Pi Camera via Picamera2 (libcamera)...")
                self.cam_obj = Picamera2()
                config_kwargs = {"main": {"size": (self.width, self.height), "format": "RGB888"}}
                if self.rotate == 180:
                    # Sensor-level flip via libcamera (no per-frame CPU cost) for an upside-down camera mount
                    from libcamera import Transform
                    config_kwargs["transform"] = Transform(hflip=1, vflip=1)
                config = self.cam_obj.create_video_configuration(**config_kwargs)
                self.cam_obj.configure(config)
                self.cam_obj.start()
                time.sleep(1.0)  # Camera warm-up
                self.active_backend = "picamera2"
                logger.info(f"✅ Picamera2 initialized successfully ({self.width}x{self.height}, rotate={self.rotate})")
                return
            except Exception as e:
                logger.warning(f"Picamera2 initialization failed: {e}")

        if self.mode in ("auto", "opencv") and OPENCV_AVAILABLE:
            try:
                logger.info("Attempting OpenCV camera capture (cv2.VideoCapture(0))...")
                self.cam_obj = cv2.VideoCapture(0)
                self.cam_obj.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cam_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if self.cam_obj.isOpened():
                    ret, frame = self.cam_obj.read()
                    if ret and frame is not None:
                        self.active_backend = "opencv"
                        logger.info(f"✅ OpenCV camera initialized successfully ({self.width}x{self.height})")
                        return
                    else:
                        self.cam_obj.release()
            except Exception as e:
                logger.warning(f"OpenCV camera initialization failed: {e}")

        # Fallback to test image file or synthetic frame
        if self.fallback_image and os.path.exists(self.fallback_image):
            self.active_backend = "file"
            logger.info(f"Using local test image stream fallback: {self.fallback_image}")
        else:
            self.active_backend = "synthetic"
            logger.info("Using in-memory synthetic frame generator fallback")

    def capture_jpeg_bytes(self) -> bytes:
        """Captures a single frame and returns encoded JPEG bytes."""
        if self.active_backend == "picamera2":
            # Picamera2 "RGB888" arrays are stored in [B, G, R] order (Picamera2 manual; measured on
            # the Pi 5 + ov5647: corr(R-B, [0]-[2]) = -0.96 against rpicam-still). Already OpenCV order.
            frame_bgr = self.cam_obj.capture_array()
            if OPENCV_AVAILABLE:
                _, jpeg_buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                return jpeg_buf.tobytes()
            else:
                # Basic PIL fallback if cv2 unavailable (PIL expects RGB)
                from PIL import Image
                img = Image.fromarray(frame_bgr[:, :, ::-1])
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=85)
                return buf.getvalue()

        elif self.active_backend == "opencv":
            ret, frame = self.cam_obj.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from OpenCV camera")
            _, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return jpeg_buf.tobytes()

        elif self.active_backend == "file":
            with open(self.fallback_image, "rb") as f:
                return f.read()

        else:
            # Synthetic solid marine cyan frame
            if OPENCV_AVAILABLE:
                import numpy as np
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                img[:, :] = (180, 100, 10)  # Deep cyan hue
                cv2.putText(img, "AquaScan Test Stream", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                _, jpeg_buf = cv2.imencode('.jpg', img)
                return jpeg_buf.tobytes()
            else:
                raise RuntimeError("No suitable capture backend available")

    def release(self):
        if self.active_backend == "picamera2" and self.cam_obj:
            try:
                self.cam_obj.stop()
            except Exception:
                pass
        elif self.active_backend == "opencv" and self.cam_obj:
            try:
                self.cam_obj.release()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="AquaScan AI Raspberry Pi Edge Camera Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="AquaScan API base URL (e.g. http://192.168.1.100:8000)")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to local image for testing fallback")
    parser.add_argument("--device-id", type=str, default="aquascan-pi-01", help="Raspberry Pi device identifier")
    parser.add_argument("--camera-id", type=str, default="picam-v2", help="Camera sensor identifier")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--interval", type=float, default=0.3, help="Transmission interval in seconds (default: 0.3s ~ 3 FPS)")
    parser.add_argument("--loop", action="store_true", help="Enable continuous streaming loop mode")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to send (0 = unlimited)")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--backend", type=str, choices=["auto", "picamera2", "opencv", "file"], default="auto", help="Camera backend selector")
    parser.add_argument("--rotate", type=int, choices=[0, 180], default=0, help="Rotate the picamera2 image (180 = upside-down mount; sensor flip, no CPU cost). Ignored by other backends")
    return parser.parse_args()


def check_server_health(server_url: str) -> bool:
    status_endpoint = f"{server_url.rstrip('/')}/api/status"
    logger.info(f"Handshaking with AquaScan server at: {status_endpoint}")
    try:
        res = requests.get(status_endpoint, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            logger.info("✅ Connection established!")
            logger.info(f"   Model       : {data.get('model')} (Fallback: {data.get('is_fallback')})")
            logger.info(f"   Device      : {data.get('device')}")
            logger.info(f"   Classes ({data.get('classes_count')}): {', '.join(data.get('classes', []))}")
            return True
        else:
            logger.error(f"Server returned non-200 status: {res.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to reach server: {e}")
        return False


def send_frame(server_url: str, jpeg_bytes: bytes, frame_id: int, args) -> bool:
    predict_endpoint = f"{server_url.rstrip('/')}/api/predict"
    
    data_payload = {
        "conf_threshold": str(args.conf),
        "source": "raspberry_pi",
        "device_id": args.device_id,
        "camera_id": args.camera_id,
        "frame_id": str(frame_id),
        "timestamp": f"{time.time():.4f}"
    }

    files_payload = {
        "file": (f"frame_{frame_id:06d}.jpg", jpeg_bytes, "image/jpeg")
    }

    t0 = time.time()
    try:
        res = requests.post(predict_endpoint, data=data_payload, files=files_payload, timeout=10.0)
        roundtrip_ms = (time.time() - t0) * 1000.0

        if res.status_code == 200:
            result = res.json()
            cnt = result.get("count", 0)
            lat = result.get("latency_ms", {}).get("total", 0.0)
            cats = result.get("category_counts", {})
            cat_str = ", ".join([f"{k}:{v}" for k, v in cats.items() if v > 0]) or "Clean Water"

            logger.info(
                f"[FRAME #{frame_id:04d}] HTTP 200 OK | Objects: {cnt} ({cat_str}) | "
                f"Inference: {lat}ms | RTT: {roundtrip_ms:.1f}ms"
            )
            return True
        else:
            logger.warning(f"[FRAME #{frame_id:04d}] Server HTTP {res.status_code}: {res.text[:100]}")
            return False

    except requests.exceptions.RequestException as e:
        logger.warning(f"[FRAME #{frame_id:04d}] Network transmission error: {e}")
        return False


def main():
    args = parse_args()
    print("=" * 65)
    print("      AquaScan AI — Raspberry Pi Edge Camera Streaming Client")
    print("=" * 65)
    print(f"Target Server : {args.server}")
    print(f"Device ID     : {args.device_id}")
    print(f"Camera ID     : {args.camera_id}")
    print(f"Resolution    : {args.width}x{args.height}")
    print(f"Mode          : {'Continuous Loop Stream' if args.loop else 'Single Frame'}")
    print(f"Interval      : {args.interval}s (~{1/args.interval:.1f} FPS)")
    print("=" * 65)

    # 1. Server Handshake
    while not check_server_health(args.server):
        logger.info("Retrying server handshake in 3 seconds...")
        time.sleep(3.0)

    # 2. Camera Initialization
    camera = CameraBackend(mode=args.backend, width=args.width, height=args.height, fallback_image=args.image, rotate=args.rotate)
    logger.info(f"Active Camera Backend: {camera.active_backend.upper()}")
    if args.rotate and camera.active_backend != "picamera2":
        logger.warning(f"--rotate {args.rotate} is only supported by the picamera2 backend; ignored for {camera.active_backend}")

    frame_counter = 0
    try:
        while True:
            frame_counter += 1
            try:
                jpeg_bytes = camera.capture_jpeg_bytes()
                send_frame(args.server, jpeg_bytes, frame_counter, args)
            except Exception as e:
                logger.error(f"Frame #{frame_counter} capture/send failure: {e}")

            if not args.loop or (args.max_frames > 0 and frame_counter >= args.max_frames):
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\nStreaming interrupted by user.")
    finally:
        camera.release()
        logger.info("Camera backend released. Edge client terminated.")


if __name__ == "__main__":
    main()
