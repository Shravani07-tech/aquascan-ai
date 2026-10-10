#!/usr/bin/env python3
"""
AquaScan AI — Plan B: on-device inference on the Raspberry Pi (headless).

Verified on device 2026-10-10 (Raspberry Pi 5 2 GB, ov5647, torch 2.14.1+cpu, ultralytics 8.4.175):
scratch_frame.jpg -> bottle 92.4% / plastic 81.5% (laptop: 92.4% / 81.8%, same boxes); 600 live frames at
imgsz 416, --threads 1 (default): 9.0 FPS, inference p50 106 ms, peak RSS 442 MB, peak 69.4 C, not throttled.
Install: see PI_SETUP.md "Plan B" (offline wheels; CPU-only torch from download.pytorch.org/whl/cpu).

Pi camera (Picamera2) -> YOLO (best.pt, or an NCNN export) on the Pi itself -> terminal output
(detections, inference time, rolling FPS) and annotated JPEGs saved on the Pi. No display windows.

Annotated frames are written to <save-dir>/latest.jpg every frame and to a fixed ring of
<save-dir>/frame_00.jpg .. frame_<ring-1>.jpg every --save-every frames, so disk use stays bounded.
A summary (FPS, inference p50/p90, process RAM, CPU temperature) is printed at the end.

Usage (on the Pi, inside ~/aquascan, venv created with --system-site-packages):
  .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --rotate 180 --frames 100
  .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --image scratch_frame.jpg   # still image
  .venv/bin/python hardware/pi_edge_infer.py --model models/best_ncnn_model --frames 0          # 0 = until Ctrl+C
"""
import argparse
import json
import os
import threading
import time
from collections import deque


def parse_args():
    p = argparse.ArgumentParser(description="AquaScan AI on-device (Raspberry Pi) inference")
    p.add_argument("--model", default="models/best.pt", help="best.pt or an exported *_ncnn_model directory")
    p.add_argument("--imgsz", type=int, default=416, help="YOLO inference size (416 = laptop webcam setting)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--frames", type=int, default=100, help="Frames to process (0 = until Ctrl+C)")
    p.add_argument("--image", default=None, help="Run on this still image instead of the camera (sanity check)")
    p.add_argument("--rotate", type=int, choices=[0, 180], default=0, help="180 = upside-down camera mount (sensor flip)")
    # Default 1: the fastest setting that stays under 70 C sustained on a Pi 5 without a fan (600 frames:
    # 1 thread 9.0 FPS / 69.4 C peak; 2 threads 9.5 FPS / 74.9 C; 4 threads 10.4 FPS / 78.7 C)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (default 1; 0 = Ultralytics default, cores - 1)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--server", default=None, help="Edge mode: POST frames + detections to <server>/api/edge/ingest (e.g. http://172.24.21.133:8000)")
    p.add_argument("--send-fps", type=float, default=5.0, help="Max frames/s sent to --server (inference still runs at full speed)")
    p.add_argument("--send-quality", type=int, default=70, help="JPEG quality of frames sent to --server")
    p.add_argument("--log-every", type=int, default=1, help="Print one per-frame line every N frames (service uses 50)")
    p.add_argument("--device-id", default="aquascan-pi5")
    p.add_argument("--camera-id", default="ov5647")
    p.add_argument("--save-dir", default="edge_output", help="Where annotated frames are written")
    p.add_argument("--save-every", type=int, default=5, help="Also save every Nth frame into the ring (0 = off)")
    p.add_argument("--ring", type=int, default=20, help="Number of ring files to cycle through")
    return p.parse_args()


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000
    except (OSError, ValueError):
        return float("nan")


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


class EdgeSender:
    """Background sender for edge mode. offer() never blocks: it drops the frame into a 1-slot mailbox
    (replacing an unsent one). The thread JPEG-encodes and POSTs it; network errors back off (1 s -> 30 s)
    and are counted, never raised, so the network can neither slow down nor stop inference."""

    def __init__(self, server, quality):
        import cv2
        import requests
        self.cv2, self.requests = cv2, requests
        self.url = server.rstrip("/") + "/api/edge/ingest"
        self.quality = quality
        self.session = requests.Session()
        self.lock, self.wake = threading.Lock(), threading.Event()
        self.slot = None
        self.sent = self.failed = self.replaced = self.skipped_backoff = 0
        self.bytes_sent, self.rtts = 0, []
        self.backoff, self.next_try, self.down_since = 0.0, 0.0, None
        threading.Thread(target=self._run, daemon=True).start()

    def offer(self, frame_bgr, meta):
        with self.lock:
            if self.slot is not None:
                self.replaced += 1
            self.slot = (frame_bgr, meta)
        self.wake.set()

    def _run(self):
        while True:
            self.wake.wait(0.5)
            self.wake.clear()
            with self.lock:
                item, self.slot = self.slot, None
            if item is None:
                continue
            if time.time() < self.next_try:
                self.skipped_backoff += 1  # server down: drop this frame, retry after the backoff
                continue
            frame, meta = item
            try:
                ok, buf = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality])
                data = buf.tobytes()
                t0 = time.time()
                r = self.session.post(self.url, files={"file": ("frame.jpg", data, "image/jpeg")},
                                      data={"meta": json.dumps(meta)}, timeout=2.0)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
                self.rtts.append((time.time() - t0) * 1000)
                self.sent += 1
                self.bytes_sent += len(data)
                if self.down_since is not None:
                    print(f"[edge] server reachable again after {time.time() - self.down_since:.0f}s")
                self.backoff, self.down_since = 0.0, None
            except Exception as e:  # network or server error: count, back off, keep going
                self.failed += 1
                self.backoff = min(max(1.0, self.backoff * 2), 30.0)
                self.next_try = time.time() + self.backoff
                if self.down_since is None:
                    self.down_since = time.time()
                    print(f"[edge] send failed ({type(e).__name__}: {str(e)[:100]}); inference continues, retrying with backoff")

    def summary(self, elapsed):
        n = self.sent
        return (f"EDGE SEND: sent {n} | failed {self.failed} | replaced-unsent {self.replaced} | dropped-during-backoff "
                f"{self.skipped_backoff} | {self.bytes_sent / max(n, 1) / 1024:.1f} KB/frame | "
                f"{self.bytes_sent / max(elapsed, 1e-9) / 1e6:.3f} MB/s | send RTT p50 {pct(self.rtts, 0.5):.0f} ms "
                f"p90 {pct(self.rtts, 0.9):.0f} ms")


def describe(model, r):
    dets = [(model.names[int(c)], float(s), [round(float(v), 1) for v in xy])
            for c, s, xy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    return dets, ", ".join(f"{n}:{s * 100:.1f}% {b}" for n, s, b in dets) or "none"


def main():
    args = parse_args()
    import cv2
    import torch
    from ultralytics import YOLO

    def apply_threads():
        # Ultralytics' select_device() resets torch to (cores - 1) threads during the first predict(),
        # so the requested thread count must be applied after the warm-up inference.
        if args.threads > 0:
            torch.set_num_threads(args.threads)
        print(f"torch CPU threads: {torch.get_num_threads()}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Loading model: {args.model}")
    t = time.time()
    model = YOLO(args.model, task="detect")
    print(f"Model loaded in {time.time() - t:.1f}s | classes: {model.names} | RSS {rss_mb():.0f} MB")

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)  # warm-up
        apply_threads()
        times = []
        for _ in range(5):
            t0 = time.time()
            r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            times.append((time.time() - t0) * 1000)
        dets, det_str = describe(model, r)
        print(f"IMAGE {args.image} {img.shape[1]}x{img.shape[0]} imgsz={args.imgsz} conf={args.conf}: {det_str}")
        print(f"inference ms p50 {pct(times, 0.5):.0f} (5 runs) | RSS {rss_mb():.0f} MB | CPU {cpu_temp_c():.1f} C")
        cv2.imwrite(os.path.join(args.save_dir, "image_result.jpg"), r.plot())
        return

    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = {"main": {"size": (args.width, args.height), "format": "RGB888"}}
    if args.rotate == 180:
        from libcamera import Transform
        cfg["transform"] = Transform(hflip=1, vflip=1)
    cam.configure(cam.create_video_configuration(**cfg))
    cam.start()
    time.sleep(1.0)  # auto-exposure settle

    # Warm-up (first inference is slow: graph build / allocation)
    model.predict(cam.capture_array(), imgsz=args.imgsz, conf=args.conf, verbose=False)
    apply_threads()

    sender = EdgeSender(args.server, args.send_quality) if args.server else None
    send_interval = 1.0 / args.send_fps if args.send_fps > 0 else 0.0
    last_offer, throttled = 0.0, 0

    recent = deque(maxlen=10)
    infer_times, n, peak_rss = [], 0, rss_mb()
    t_start = t_prev = time.time()
    print(f"Running on-device inference (imgsz={args.imgsz}, conf={args.conf}, rotate={args.rotate}). Ctrl+C to stop.")
    if sender:
        print(f"Edge mode: sending to {sender.url} at <= {args.send_fps} fps, JPEG quality {args.send_quality}")
    try:
        while args.frames == 0 or n < args.frames:
            # Picamera2 RGB888 arrays are [B, G, R] (verified on this Pi), which is what YOLO expects for numpy input
            captured_at = time.time()
            frame_bgr = cam.capture_array()
            t0 = time.time()
            r = model.predict(frame_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            infer_ms = (time.time() - t0) * 1000
            infer_times.append(infer_ms)
            n += 1

            now = time.time()
            recent.append(now - t_prev)
            t_prev = now
            fps = len(recent) / sum(recent)
            peak_rss = max(peak_rss, rss_mb())

            dets, det_str = describe(model, r)
            if args.log_every <= 1 or n % args.log_every == 0:
                print(f"[{n:04d}] {len(dets)} obj ({det_str}) | infer {infer_ms:.0f} ms | {fps:.2f} FPS"
                      + (f" | sent {sender.sent} failed {sender.failed}" if sender else ""))

            if sender:
                if now - last_offer >= send_interval:
                    last_offer = now
                    sender.offer(frame_bgr, {
                        "device_id": args.device_id, "camera_id": args.camera_id, "frame_id": n,
                        "captured_at": captured_at, "imgsz": args.imgsz, "conf": args.conf,
                        "inference_ms": round(infer_ms, 1), "fps": round(fps, 2), "cpu_temp_c": cpu_temp_c(),
                        "detections": [{"class": c, "confidence": round(s, 4), "bbox": b} for c, s, b in dets],
                    })
                else:
                    throttled += 1

            annotated = r.plot()  # BGR with boxes and labels
            cv2.imwrite(os.path.join(args.save_dir, "latest.jpg"), annotated)
            if args.save_every and n % args.save_every == 0:
                slot = (n // args.save_every - 1) % args.ring
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{slot:02d}.jpg"), annotated)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cam.stop()
        cam.close()
        elapsed = time.time() - t_start
        if n:
            print(f"SUMMARY: {n} frames in {elapsed:.1f}s = {n / elapsed:.2f} FPS overall | "
                  f"inference ms p50 {pct(infer_times, 0.5):.0f} p90 {pct(infer_times, 0.9):.0f} | "
                  f"peak RSS {peak_rss:.0f} MB | CPU {cpu_temp_c():.1f} C | threads {torch.get_num_threads()}")
        if sender:
            time.sleep(0.5)  # let an in-flight send finish before reporting
            print(sender.summary(elapsed) + f" | throttled-by-send-fps {throttled}")
        print(f"Annotated frames in: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
