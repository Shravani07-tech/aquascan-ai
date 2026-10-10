"""
AquaScan AI — Raspberry Pi Stream Cache Service
Stores in-memory cache of the latest frame and detection result received from Raspberry Pi edge client.
"""

import time
from threading import Lock

class RPiStreamStore:
    def __init__(self):
        self._lock = Lock()
        self.latest_image_bytes: bytes = None
        self.latest_prediction: dict = None
        self.latest_metadata: dict = None
        self.last_updated: float = 0.0

    def update(self, image_bytes: bytes, prediction: dict, metadata: dict):
        with self._lock:
            self.latest_image_bytes = image_bytes
            self.latest_prediction = prediction
            self.latest_metadata = metadata or {}
            self.last_updated = time.time()

    def get_latest_state(self, max_age_seconds: float = 5.0) -> dict:
        with self._lock:
            if not self.latest_image_bytes or not self.latest_prediction:
                return {
                    "success": True,
                    "active": False,
                    "message": "No Raspberry Pi frame received yet",
                    "hardware_metadata": None,
                    "prediction": None,
                    "last_updated": self.last_updated
                }

            age = time.time() - self.last_updated
            is_active = age <= max_age_seconds
            return {
                "success": True,
                "active": is_active,
                "age_seconds": round(age, 2),
                "hardware_metadata": self.latest_metadata,
                "prediction": self.latest_prediction,
                "last_updated": self.last_updated,
                "image_url": f"/api/rpi/latest/image?t={int(self.last_updated * 1000)}"
            }

    def get_latest_image_bytes(self) -> bytes:
        with self._lock:
            return self.latest_image_bytes

rpi_stream_store = RPiStreamStore()
