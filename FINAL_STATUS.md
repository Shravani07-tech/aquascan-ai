# AquaScan AI — Final Status (2026-10-10, tag `release-2026-10-10`)

## What works (verified on real hardware)
- **Laptop app**: FastAPI backend + browser UI; Laptop Webcam / Raspberry Pi / Upload-Demo modes. Full
  TESTING.md regression **PASS** on `09b3fc2` (2026-10-10 12:25, laptop on AC): #0–#15 and #17 all pass (see Regression).
- **Raspberry Pi — Plan A** (Pi camera → laptop inference → dashboard): Raspberry Pi 5 + ov5647 →
  `hardware/rpi_client.py --backend picamera2 --rotate 180` → laptop backend `:8000` → `best.pt` → dashboard
  **Raspberry Pi** panel shows the live frame with boxes. **3.8 fps**, 120/120 HTTP 200, round trip p50 119 ms.
- **Raspberry Pi — Plan B** (inference ON the Pi): `hardware/pi_edge_infer.py` runs `best.pt` on the Pi 5 CPU.
  **1 thread default (9 fps sustained, ≤70 °C)**: 600 live frames, imgsz 416, 9.0 FPS, inference p50 106 ms,
  peak 69.4 °C, not throttled, ~450 MB RAM. **With a Pi 5 active cooler, `--threads 4` gives ~10 fps; re-check
  temps first** (without a fan: 2 threads 74.9 °C, 4 threads 78.7 °C peak). Same detections as the laptop:
  `scratch_frame.jpg` → bottle 92.4% / plastic 81.5% (laptop 92.4% / 81.8%, identical boxes).
  Installed offline (CPU-only torch 2.14.1+cpu, ultralytics 8.4.175) — commands in PI_SETUP.md.
- **Model integrity**: `models/best.pt` SHA-256 `d8248744d7d5ae2758f61637572f6726f60e938ceb53191be47ac66195741b78`,
  unchanged throughout (same hash on the Pi's copy). imgsz 416 for laptop webcam, 640 for Pi/upload, unchanged.

## Start commands
Laptop (PowerShell, project folder; charger in):
```
.venv\Scripts\python.exe run_server.py
```
Browser: `http://localhost:8000` (Ctrl+F5 once) → **Raspberry Pi**, **Laptop Webcam** or **Upload / Demo**.
Check IPs first (DHCP on the hotspot): laptop `ipconfig | findstr IPv4`; Pi `ssh shanki@shanky.local "hostname -I"`.

**Plan A** (Pi camera → laptop dashboard; backend must be running; laptop IP 172.24.21.133 at last check):
```
ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/rpi_client.py --server http://172.24.21.133:8000 --device-id aquascan-pi5 --camera-id ov5647 --backend picamera2 --rotate 180 --loop --interval 0.1"
```
**Plan B** (inference on the Pi, headless; prints FPS + detections, saves `~/aquascan/edge_output/latest.jpg`):
```
ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --rotate 180 --frames 0"
```
Do not run Plan A and Plan B at the same time (both open the Pi camera). Stop either with Ctrl+C.

## Fixed / added (2026-10-06 → 10)
- Windows Smart App Control blocked torchvision → pure-PyTorch NMS fallback (identical detections; 3000/3000 vs native).
- Stale cached `app.js` → no-cache headers. Camera Occupied message names the terminal client.
- **Webcam confidence**: the slider was silently capped at 20% in webcam mode. Now disabled there and labelled
  "20% (fixed in webcam mode)"; deployed threshold unchanged (0.20); Upload/Demo and Pi still honour it (`65e1533`).
- Pi camera **red/blue swap** in `rpi_client.py` picamera2 backend (`dc8c6ab`); `--rotate 180` for the
  upside-down mount, sensor flip, no CPU cost (`01fcc5c`).
- Plan B on-device script with `--threads` (Ultralytics resets torch threads on first predict; applied after warm-up).
- Debug-only frame collector (`AQUASCAN_SAVE_FRAMES=1`, off by default; off for the release regression).
- Reproducible regression harness (TESTING.md #0–#17) and test-only fixes.

## Investigated, no code change needed
- **Box positioning in laptop webcam mode**: no rendering bug. Overlay matches the displayed picture to ≤0.3 px at
  9 window sizes / display scales (#14); mirrored video draws flipped boxes correctly (#15, negative control
  fails as it should); real webcam at the user's 150% display: server box vs browser overlay within 2 px (#16).
  "Wrong-position" boxes were (a) model false positives on non-objects and (b) latency lag on battery.

## Open items
1. **imgsz is UNVALIDATED**: the 416 evidence frame was a false positive on a person. Current sizes: webcam 416,
   Plan A 640, Plan B 416. Next water-scene test must compare 416 vs 640 on the same saved frames (detection
   rate, FP rate, Pi fps).
2. **Water-scene detection test** (v2 §17: ≥60% detection, ≤0.10 FP/frame): pending — not allowed in the lab.
   Use the frame collector (`AQUASCAN_SAVE_FRAMES=1`) so item 1 can be answered from the same frames.

## Known limitations
- **Indoor false positives**: the model is trained on water scenes (outdoor floating waste, low oblique view).
  Indoors it fires on ceiling lights, tube lights, logos, faces and shirts (e.g. "bottle" 86% on a ceiling
  light, "plastic" 29% on tube lights). **For demos the camera must look down at water/floor, never at the
  ceiling.** Hand-held bottles are generally not detected.
- **Water-scene detection test: pending (not allowed in lab).** v2 §17 detection-rate (≥60%) and false-positive
  (≤0.10/frame) thresholds have not been measured on a valid in-domain scene.
- Model trained for 10 epochs (mAP50 ≈69.4%, recall ≈64.7%): interim metrics; dataset_webcam is empty.
- **imgsz=416 for the laptop webcam has no valid supporting evidence.** The handoff's only evidence ("bottle 30.50%
  at [0.5, 287.1, 141.6, 480.0]" on the old reference frame) was checked on 2026-10-10: that frame contains no
  bottle; the box is on a person's shoulder/shirt (a false positive). 416 stays deployed (not proven wrong either);
  re-decide with the offline imgsz sweep on real water-scene frames.
- Laptop on battery throttles the CPU (inference 400–1000 ms vs ~60–90 ms on AC): keep the charger in.
- Pi 5 without a fan: keep `--threads 1` (default) for sustained runs.

## Regression (`09b3fc2`, 2026-10-10 12:25, AC power, collector off — 887 debug frames before = after)
| # | Result |
|---|---|
| 0–7 | PASS — JS syntax; API 13/13; Pi streaming 10/10 (sim); hw sim; real detection; laptop E2E; dataset 7644/21111 valid; restart cycles 25/25 |
| 8 / 8b | PASS — 58/58 and 62/62 HTTP 200, 5/5 cycles each, avg 89 / 98 ms, 0 console errors |
| 9–11 | PASS — rendering 28/28 BOTTLE 92.6%; camera busy → Try Again 12/12; ended-track recovery |
| 12 | PASS — 300/300 HTTP 200 |
| 13 | PASS — Pi panel ONLINE (sim); Upload/Demo 6/6, every count matches the API |
| 14 | PASS — geometry ≤0.3 px at 1275×800@1.5, 1920×1080, 700×800, 400×800@2 |
| 15 | PASS — mirrored overlay x 139–267 vs expected 141–266 (2 viewports) |
| 16 | diagnostic, run earlier today: PASS (2 px) — needs an object in view, not in the automated run |
| 17 | PASS — webcam fixed 20% (sends 0.2); Upload honours slider (sends 0.3) |
| Hardware | Plan A from the real Pi 30/30 HTTP 200; Plan B on the Pi: bottle 92.4% / plastic 81.5% |
