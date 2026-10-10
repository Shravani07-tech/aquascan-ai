# AquaScan AI — Raspberry Pi Setup (Plan A: Pi camera → laptop inference)

**Status (verified 2026-10-09 10:18–10:19):** real Raspberry Pi 5 + Pi camera → `hardware/rpi_client.py`
(`--backend picamera2`) → laptop backend `:8000` → `best.pt` → dashboard **Raspberry Pi** panel shows the
live Pi frame with boxes. 120/120 frames HTTP 200, **3.8 fps** at `--interval 0.1`, round trip p50 119 ms /
p90 193 ms, server inference p50 55 ms. Pi at 46 °C, not throttled. Proof: `docs/pi_plan_a_dashboard.jpg`
(the box in that screenshot is a false positive on a ceiling light: the camera was pointing at the ceiling).

## Hardware and network (as verified)

| Item | Value |
|---|---|
| Pi | Raspberry Pi 5, 2 GB RAM, `aarch64`, Debian 13 (trixie) Raspberry Pi OS, Python 3.13.5 |
| Camera | Pi camera module **ov5647** (ribbon cable), 640×480 @ up to 62.5 fps |
| Pi user / IP | `shanki` @ **172.24.21.132** (phone hotspot, DHCP — may change) |
| Laptop IP | **172.24.21.133** (Wi-Fi, same /24) — DHCP, re-check each session |
| Swap | 2 GiB zram (`/dev/zram0`, `rpi-swap`) — already present |
| Pi Python deps | picamera2, requests, PIL, numpy: all system packages (apt). **No OpenCV** on the Pi: the client uses its PIL encoder |

## 1. Find the laptop's IP (every session — the hotspot hands out addresses by DHCP)

On the laptop (PowerShell):
```
Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias Wi-Fi | Select-Object IPAddress
```
Use the address on the same `x.y.z.*` subnet as the Pi. The backend also prints a "LAN / Raspberry Pi URL" at startup.

## 2. Start the laptop backend

```
.venv\Scripts\python.exe run_server.py
```
Wait for `AquaScan Trained Model loaded: best.pt (5 classes ...)` and `Uvicorn running on http://0.0.0.0:8000`.
Do **not** set `AQUASCAN_SAVE_FRAMES` for the demo.

## 3. Firewall (laptop) — normally nothing to do

Windows treats the hotspot as a **Public** network. Inbound TCP 8000 already works because Windows has
enabled **Allow** rules for `...\Python312\python.exe` on the Public profile (the backend's real process).
Verify from the Pi (step 6). Only if the Pi times out, an administrator PowerShell can add:
```
New-NetFirewallRule -DisplayName "AquaScan backend 8000 (LAN)" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress LocalSubnet -Action Allow -Profile Public,Private
```
(`ping` from the Pi to the laptop fails even when HTTP works: Windows blocks ICMP by default. Test with curl, not ping.)

## 4. SSH from the laptop (key login is set up)

```
ssh shanki@172.24.21.132
```
The laptop key is `%USERPROFILE%\.ssh\id_ed25519`; its public half is in the Pi's `~/.ssh/authorized_keys`.

## 5. Install / update the client on the Pi (from the laptop, project folder)

```
ssh shanki@172.24.21.132 "mkdir -p ~/aquascan/hardware"
scp hardware/rpi_client.py shanki@172.24.21.132:aquascan/hardware/rpi_client.py
ssh shanki@172.24.21.132 "cd ~/aquascan && python3 -m venv --system-site-packages .venv"
```
`--system-site-packages` is required: Picamera2 is installed by apt and cannot be pip-installed cleanly.
The client must include commit `dc8c6ab` (picamera2 colour fix) — older copies send red/blue swapped frames.
Check on the Pi: `grep -c "frame_bgr\[:, :, ::-1\]" ~/aquascan/hardware/rpi_client.py` must print `1`.
If in doubt, just re-run the `scp` line above (hash comparisons can differ only by Windows line endings).

## 6. 2-minute smoke test

1. Laptop: backend running (step 2). Browser: `http://localhost:8000` → **Raspberry Pi** tab.
2. Pi reaches the backend:
   ```
   ssh shanki@172.24.21.132 "curl -s -m 5 http://172.24.21.133:8000/api/status | head -c 80"
   ```
   Expect `{"success":true,"model":"AquaScan YOLOv8n"...`.
3. Pi camera is visible:
   ```
   ssh shanki@172.24.21.132 "rpicam-hello --list-cameras"
   ```
   Expect `0 : ov5647 ...`.
4. Stream 30 frames:
   ```
   ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/rpi_client.py --server http://172.24.21.133:8000 --device-id aquascan-pi5 --camera-id ov5647 --backend picamera2 --rotate 180 --loop --max-frames 30 --interval 0.1"
   ```
   Expect 30 lines `HTTP 200 OK`, and the dashboard panel **ONLINE** with Device `aquascan-pi5`, Camera `ov5647`,
   a live frame and its Frame ID increasing.

**`--rotate 180`:** the Pi camera on this rig is mounted upside down (on the dashboard the ceiling showed
at the bottom and label text was inverted). The flag flips the image on the sensor via libcamera
`Transform(hflip=1, vflip=1)`: verified as an exact 180° rotation (corr +0.97) at the same 33 ms/frame
capture cost as without it. Drop the flag if the camera is remounted upright. It only affects
`--backend picamera2`; other backends ignore it with a warning.

## 7. Demo run (continuous)

```
ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/rpi_client.py --server http://172.24.21.133:8000 --device-id aquascan-pi5 --camera-id ov5647 --backend picamera2 --rotate 180 --loop --interval 0.1"
```
Stop with **Ctrl+C**. The stream stops if the SSH session closes; to keep it running after disconnecting:
```
ssh shanki@172.24.21.132 "cd ~/aquascan && nohup .venv/bin/python hardware/rpi_client.py --server http://172.24.21.133:8000 --device-id aquascan-pi5 --camera-id ov5647 --backend picamera2 --rotate 180 --loop --interval 0.1 > ~/aquascan/client.log 2>&1 &"
ssh shanki@172.24.21.132 "pkill -f rpi_client.py"
```
The dashboard Pi panel goes OFFLINE 5 s after the last frame.

**Detection quality:** the model is trained on floating waste seen from water level. Point the Pi camera at
an in-domain scene (bottle floating in a tub of water, low oblique angle, or a phone/screen showing a
dataset image) — not at a ceiling or a hand-held bottle.

## Failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to reach server` / curl from Pi times out | Wrong laptop IP (hotspot DHCP changed), backend not running, or firewall | Re-check step 1 IP, step 2 backend, then step 3 rule |
| Laptop IP not on the Pi's subnet | Laptop on a different Wi-Fi | Connect both to the same hotspot |
| `ssh: connect ... timed out` | Pi IP changed or Pi off | Check the phone hotspot's connected-devices list for the Pi's new IP |
| `rpicam-hello --list-cameras` shows nothing | Ribbon cable loose/reversed | Power **off** the Pi, reseat the cable (contacts facing the connector's contacts), power on. `camera_auto_detect=1` is already set in `/boot/firmware/config.txt` |
| `Device or resource busy` / Picamera2 fails to start | Another process holds the camera (e.g. a leftover `rpicam-*` or client) | `pkill -f rpicam; pkill -f rpi_client.py`, retry |
| `ModuleNotFoundError: picamera2` | venv created without system packages | `rm -rf ~/aquascan/.venv` then recreate with `--system-site-packages` (step 5) |
| Dashboard image upside down (ceiling at the bottom, text inverted) | Camera mounted upside down | Add `--rotate 180` (or remove it if the camera was remounted upright). The model was trained on upright scenes |
| Colours look swapped (red↔blue) on the dashboard | Old client without commit `dc8c6ab` | Re-copy `hardware/rpi_client.py` (step 5) |
| Client falls back to `synthetic`/`file` backend | Picamera2 init failed | Read the warning line above `Active Camera Backend`, then the camera rows above |
| Dashboard Pi panel OFFLINE while client prints HTTP 200 | Browser on a different tab/source, or >5 s gap | Select the **Raspberry Pi** source; check client is still sending |
| Very slow frames (> 1 s RTT) | Laptop on battery (CPU throttled to ~50%) | Plug in the charger |
| `HTTP 500 ... torchvision::nms` | Smart App Control blocked torchvision on the laptop | Already handled by the NMS fallback in `backend/services/inference.py`; restart the backend |

## Plan B: run best.pt ON the Pi (verified 2026-10-10)

`hardware/pi_edge_infer.py`: Pi camera → YOLO `best.pt` on the Pi itself → detections, inference time and FPS
in the terminal; annotated frames saved on the Pi (`edge_output/latest.jpg` every frame, plus a ring of 20
`frame_NN.jpg`). Headless, no display needed. Proof: `docs/pi_plan_b_proof.jpg`.

| Result (Pi 5 2 GB, no fan, imgsz 416, conf 0.25) | |
|---|---|
| Sanity check `scratch_frame.jpg` | bottle **92.4%** / plastic **81.5%** (laptop 92.4% / 81.8%, identical boxes) |
| Sustained 600 live frames, `--threads 1` (default) | **9.0 FPS**, inference p50 **106 ms**, peak **69.4 °C**, not throttled |
| Memory | process peak ~450 MB, system peak 754 of 2010 MiB, no swap |

**`--threads` (torch CPU threads), sustained 600-frame runs from < 50 °C:**

| Threads | FPS | Peak temp | |
|---|---|---|---|
| **1 (default)** | **9.0** | **69.4 °C** | fastest setting under 70 °C |
| 2 | 9.5 | 74.9 °C | over 70 °C |
| 4 | 10.4 | 78.7 °C | over 70 °C |

Ultralytics resets torch to (cores − 1) = 3 threads during its first `predict()`, so the script applies `--threads`
after the warm-up inference. With an active cooler, `--threads 4` would gain ~15 % speed. NCNN export was not
needed (`.pt` already well above 3 FPS). (The very first run this morning measured 6.4 FPS at the default 3
threads; every later run at 3 threads gave ~9.7 FPS — the slower first run was not explained.)

### Install (offline — the hotspot drops large downloads)
Done once; already installed in `~/aquascan/.venv` (torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.175).
To reinstall from scratch, on the **laptop** (project folder, PowerShell):
```
.venv\Scripts\python.exe -m pip download --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 --python-version 3.13 --implementation cp --only-binary=:all: -d pi_wheels ultralytics torch torchvision
.venv\Scripts\python.exe -m pip download --index-url https://download.pytorch.org/whl/cpu --platform manylinux_2_28_aarch64 --python-version 3.13 --implementation cp --only-binary=:all: --no-deps -d pi_wheels_cpu torch==2.14.1 torchvision==0.29.1
scp -r pi_wheels pi_wheels_cpu shanki@172.24.21.132:
```
Then on the Pi:
```
cd ~/aquascan && .venv/bin/pip install --no-index --find-links ~/pi_wheels_cpu --find-links ~/pi_wheels "torch==2.14.1+cpu" "torchvision==0.29.1+cpu" ultralytics
```
**Why the second download:** PyPI's aarch64 torch is a CUDA build that requires `cuda-toolkit` on Linux. The
laptop's `pip download` evaluates that condition for Windows and skips it, so the offline install fails with
`No matching distribution found for cuda-toolkit`. The `+cpu` builds have no CUDA dependency (and are 159 MB
instead of 454 MB). A harmless pip warning about `types-seaborn` / `pandas-stubs` can be ignored.

### Run (on the Pi, over SSH)
```
ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --image scratch_frame.jpg"
ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --rotate 180 --frames 100"
ssh shanki@172.24.21.132 "cd ~/aquascan && .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --rotate 180 --frames 0"
```
(sanity check / 100-frame speed test / continuous until Ctrl+C). Copy the newest annotated frame to the laptop:
`scp shanki@172.24.21.132:aquascan/edge_output/latest.jpg .`
Do not run Plan A (`rpi_client.py`) and Plan B at the same time: both open the Pi camera.

## What to say about the Pi

"The Raspberry Pi 5 runs the same trained model on its own CPU at about 9 frames per second (Plan B), and can
also stream its camera to the AquaScan server for detection on the laptop (Plan A)."
