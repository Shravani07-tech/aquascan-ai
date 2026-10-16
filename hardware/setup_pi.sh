#!/usr/bin/env bash
# AquaScan AI — one-command Raspberry Pi setup. Run ON THE PI from the cloned repo:
#     cd ~/aquascan && bash hardware/setup_pi.sh
# Safe to re-run: every step checks first and skips what is already done. Stops with a clear message on failure.
#
# torch/ultralytics come from (first match):
#   1. offline wheels in ~/pi_wheels_cpu + ~/pi_wheels (copied from the laptop; Python 3.13 only), or
#   2. online: CPU-only torch from download.pytorch.org/whl/cpu, then ultralytics from PyPI.
# Never installs PyPI's aarch64 torch: that is a CUDA build whose cuda-toolkit dependency breaks on a Pi.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
EXPECTED_SHA="d8248744d7d5ae2758f61637572f6726f60e938ceb53191be47ac66195741b78"
TORCH_VER="2.14.1"
TV_VER="0.29.1"
WHEELS="${AQUASCAN_WHEELS:-$HOME/pi_wheels}"
WHEELS_CPU="${AQUASCAN_WHEELS_CPU:-$HOME/pi_wheels_cpu}"
PY=".venv/bin/python"

step() { echo; echo "=== $*"; }
ok()   { echo "  [OK] $*"; }
warn() { echo "  [WARN] $*"; }
fail() { echo; echo "  [FAIL] $*" >&2; exit 1; }

step "1/6 System checks"
[ "$(uname -m)" = "aarch64" ] || fail "Needs 64-bit Raspberry Pi OS (aarch64); this is $(uname -m)."
ok "64-bit OS ($(uname -m))"
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "Python $PYV"
python3 -c "import picamera2" 2>/dev/null || fail "python3-picamera2 is missing: sudo apt install -y python3-picamera2"
ok "picamera2 (system package)"
python3 -c "import requests" 2>/dev/null || fail "python3-requests is missing: sudo apt install -y python3-requests"
ok "requests (system package)"
if command -v rpicam-hello >/dev/null && rpicam-hello --list-cameras 2>&1 | grep -q '^[0-9] :'; then
    ok "camera: $(rpicam-hello --list-cameras 2>&1 | grep '^[0-9] :' | head -1 | cut -c1-60)"
else
    warn "no camera detected (check the ribbon cable with the Pi powered off). Setup continues; camera tests will not run."
fi

step "2/6 Model file"
[ -f models/best.pt ] || fail "models/best.pt not found. Clone the full repo: git clone git@github.com:Shravani07-tech/aquascan-ai.git ~/aquascan"
SHA="$(sha256sum models/best.pt | cut -d' ' -f1)"
[ "$SHA" = "$EXPECTED_SHA" ] || fail "models/best.pt hash mismatch ($SHA). The file is incomplete or wrong: delete the folder and clone again."
ok "models/best.pt SHA-256 matches ($(stat -c %s models/best.pt) bytes)"

step "3/6 Python environment (.venv, with system packages so picamera2 stays visible)"
if [ ! -x "$PY" ]; then
    python3 -m venv --system-site-packages .venv
    ok "created .venv"
else
    ok ".venv exists"
fi
"$PY" -c "import picamera2" 2>/dev/null || fail ".venv cannot see picamera2 (created without --system-site-packages). Run: rm -rf .venv  and re-run this script."
ok ".venv sees picamera2"

step "4/6 torch + ultralytics (CPU-only)"
if "$PY" -c "import torch, torchvision, ultralytics" 2>/dev/null; then
    ok "already installed"
elif [ "$PYV" = "3.13" ] && [ -d "$WHEELS_CPU" ] && [ -d "$WHEELS" ]; then
    echo "  installing OFFLINE from $WHEELS_CPU + $WHEELS"
    "$PY" -m pip install --no-index --find-links "$WHEELS_CPU" --find-links "$WHEELS" \
        "torch==${TORCH_VER}+cpu" "torchvision==${TV_VER}+cpu" ultralytics
else
    [ -d "$WHEELS" ] && warn "offline wheels are for Python 3.13 but this Pi has $PYV: installing online instead"
    echo "  installing ONLINE (CPU-only torch from download.pytorch.org; about 200 MB)"
    "$PY" -m pip install --resume-retries 20 --timeout 60 --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
      || "$PY" -m pip install --resume-retries 20 --timeout 60 --index-url https://download.pytorch.org/whl/cpu torch torchvision
    "$PY" -m pip install --resume-retries 20 --timeout 60 ultralytics
fi
"$PY" - <<'EOF' || fail "torch/ultralytics check failed (see above)"
import torch, torchvision, ultralytics, cv2
assert not torch.cuda.is_available()
print(f"  [OK] torch {torch.__version__} | torchvision {torchvision.__version__} | ultralytics {ultralytics.__version__} | opencv {cv2.__version__} | CUDA: no (CPU build)")
EOF

step "5/6 Sanity check: best.pt on scratch_frame.jpg (expected bottle ~92.4%, plastic ~81.5%)"
OUT="$("$PY" hardware/pi_edge_infer.py --model models/best.pt --image scratch_frame.jpg 2>&1 | grep -E '^IMAGE|inference ms' || true)"
echo "$OUT" | sed 's/^/  /'
echo "$OUT" | grep -qE 'bottle:9[0-4]\.[0-9]%' || fail "sanity check did not find the bottle at ~92%: the model is not running correctly."
ok "model detects the bottle on the Pi like on the laptop"

step "6/6 Done"
cat <<EOF
  Everything needed to run the model on this Pi is installed.
  Plan B (model on the Pi, terminal output):
    .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --rotate 180 --frames 100
  Edge mode (model on the Pi, results on the laptop dashboard):
    .venv/bin/python hardware/pi_edge_infer.py --model models/best.pt --rotate 180 --frames 0 --server http://<LAPTOP_IP>:8000
  Plan A (camera on the Pi, model on the laptop):
    .venv/bin/python hardware/rpi_client.py --server http://<LAPTOP_IP>:8000 --device-id aquascan-pi5 --camera-id ov5647 --backend picamera2 --rotate 180 --loop --interval 0.1
  Auto-start edge mode on boot: bash hardware/install_edge_service.sh http://<LAPTOP_IP>:8000
  (--rotate 180 is for an upside-down camera mount; drop it if the camera is mounted upright.)
EOF
