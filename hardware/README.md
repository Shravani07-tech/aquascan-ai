# 🔌 AquaScan AI — Raspberry Pi & Camera Hardware Integration Guide

This guide provides step-by-step instructions for configuring and running the **Raspberry Pi Edge Node** to capture real-time camera frames and stream them over LAN to the central **AquaScan AI FastAPI Inference Server**.

---

## 📋 System Requirements

### 1. Hardware Requirements
- **Raspberry Pi**: Raspberry Pi 4 Model B, Raspberry Pi 5, or Raspberry Pi 3B+.
- **Camera Module**: Raspberry Pi Camera Module v1/v2/v3 (Ribbon cable) OR standard USB Webcam.
- **Network**: Wi-Fi or Ethernet connection to the same local area network (LAN) as the host PC running AquaScan AI.

### 2. Software Requirements
- **Raspberry Pi OS**: Bookworm (64-bit or 32-bit recommended) or Bullseye.
- **Python Version**: Python 3.8 or higher.

---

## 🛠️ Step 1 — Raspberry Pi Setup & Installation

Log in to your Raspberry Pi via terminal/SSH and run:

```bash
# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Install Raspberry Pi Camera libcamera & Python dependencies
sudo apt install -y python3-picamera2 python3-requests python3-opencv

# 3. Verify camera detection (for Pi Ribbon Camera)
libcamera-hello
```

---

## 🌐 Step 2 — Network Setup & Finding the Server LAN IP

1. **Start the AquaScan Server on your Host PC**:
   On your main computer running AquaScan AI, execute:
   ```cmd
   python run_server.py
   ```
   Look for the output line:
   ```text
   LAN / Raspberry Pi URL: http://192.168.1.105:8000
   ```
   *(Note down your actual LAN IP address, e.g. `192.168.1.105`).*

2. **Configure Windows Defender Firewall (Important)**:
   If your Raspberry Pi cannot connect to port 8000:
   - Open **Windows Defender Firewall** -> **Advanced Settings**.
   - Select **Inbound Rules** -> **New Rule**.
   - Rule Type: **Port** -> TCP -> Specific local ports: `8000`.
   - Action: **Allow the connection** (Private Networks).

3. **Test Network Connectivity from Raspberry Pi**:
   From your Raspberry Pi terminal, test reachability:
   ```bash
   curl http://<YOUR_HOST_LAN_IP>:8000/api/status
   ```
   *(Expected response: `{"success":true,"model":"AquaScan YOLOv8n",...}`)*

---

## 🚀 Step 3 — Running the Edge Camera Client

Clone or copy the `hardware/rpi_client.py` file to your Raspberry Pi:

### 1. Standard Continuous Stream (Pi Camera, ~3 FPS):
```bash
python3 rpi_client.py --server http://<YOUR_HOST_LAN_IP>:8000 --loop --interval 0.3
```

### 2. High-Frequency Continuous Stream (~5 FPS):
```bash
python3 rpi_client.py --server http://<YOUR_HOST_LAN_IP>:8000 --loop --interval 0.2
```

### 3. Custom Resolution & Device Identification:
```bash
python3 rpi_client.py \
  --server http://192.168.1.105:8000 \
  --device-id aquascan-pi-lab01 \
  --camera-id picam-v2 \
  --width 640 \
  --height 480 \
  --loop \
  --interval 0.3
```

### 4. Offline / Test Mode (Using Local Image File):
If testing without a physical camera attached:
```bash
python3 rpi_client.py --server http://localhost:8000 --image test.jpg --loop --interval 0.5
```

---

## 📊 Expected Terminal Output on Raspberry Pi

When running successfully, `rpi_client.py` outputs live telemetry:

```text
=================================================================
      AquaScan AI — Raspberry Pi Edge Camera Streaming Client
=================================================================
Target Server : http://192.168.1.105:8000
Device ID     : aquascan-pi-01
Camera ID     : picam-v2
Resolution    : 640x480
Mode          : Continuous Loop Stream
Interval      : 0.3s (~3.3 FPS)
=================================================================
17:30:00 [INFO] Handshaking with AquaScan server at: http://192.168.1.105:8000/api/status
17:30:00 [INFO] ✅ Connection established!
17:30:00 [INFO]    Model       : AquaScan YOLOv8n (Fallback: False)
17:30:00 [INFO]    Device      : cpu
17:30:00 [INFO]    Classes (5) : bottle, can, carton, paper, plastic
17:30:01 [INFO] Initializing Raspberry Pi Camera via Picamera2 (libcamera)...
17:30:02 [INFO] ✅ Picamera2 initialized successfully (640x480)
17:30:02 [INFO] Active Camera Backend: PICAMERA2
17:30:02 [INFO] [FRAME #0001] HTTP 200 OK | Objects: 2 (bottle:2) | Inference: 78.5ms | RTT: 94.2ms
17:30:03 [INFO] [FRAME #0002] HTTP 200 OK | Objects: 3 (bottle:2, carton:1) | Inference: 81.2ms | RTT: 98.0ms
17:30:03 [INFO] [FRAME #0003] HTTP 200 OK | Objects: 1 (plastic:1) | Inference: 76.1ms | RTT: 91.5ms
```

---

## ❓ Troubleshooting

| Error Symptom | Cause | Solution |
|---|---|---|
| **`Connection refused` / `Max retries exceeded`** | Server not running or blocked by firewall. | Verify `run_server.py` is running on host PC. Check Windows Firewall rule for port 8000. |
| **`No module named 'picamera2'`** | Missing `python3-picamera2` package. | Run `sudo apt install python3-picamera2`. If using USB camera, use `--backend opencv`. |
| **`libcamera` device not found** | Ribbon cable loose or camera disabled. | Check physical ribbon cable connection. Run `libcamera-hello` to test sensor. |
| **High Network Latency (>500ms)** | Wi-Fi congestion or oversized frames. | Keep resolution at `640x480` and increase `--interval` to `0.5` seconds. |
