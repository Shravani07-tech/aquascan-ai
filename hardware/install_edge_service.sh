#!/usr/bin/env bash
# AquaScan AI — install the edge-mode auto-start service on the Pi (run ON THE PI, after setup_pi.sh):
#     cd ~/aquascan && bash hardware/install_edge_service.sh http://<LAPTOP_IP>:8000
# Shows the service file and settings and asks before changing anything. sudo will ask for the Pi password.
# Stop:     sudo systemctl stop aquascan-edge        (frees the camera for Plan A / manual runs)
# Disable:  sudo systemctl disable --now aquascan-edge   (no auto-start on boot)
# Logs:     journalctl -u aquascan-edge -f
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
SERVER="${1:-}"
[ -n "$SERVER" ] || { echo "Usage: bash hardware/install_edge_service.sh http://<LAPTOP_IP>:8000"; exit 1; }
[ "$REPO_DIR" = "/home/shanki/aquascan" ] || { echo "The service file expects the repo at /home/shanki/aquascan (this is $REPO_DIR)."; exit 1; }
[ -x .venv/bin/python ] && .venv/bin/python -c "import ultralytics" 2>/dev/null || { echo "Run hardware/setup_pi.sh first."; exit 1; }

if curl -s -m 5 -o /dev/null "$SERVER/api/status"; then
    echo "[OK] laptop backend reachable at $SERVER"
else
    echo "[WARN] $SERVER is not reachable right now. The service will keep inferring and retry sending until it is."
fi

ENV_FILE="$REPO_DIR/edge.env"
cat > "$ENV_FILE.new" <<EOF
# Read by aquascan-edge.service. Edit AQUASCAN_SERVER if the laptop's IP changes, then: sudo systemctl restart aquascan-edge
AQUASCAN_SERVER=$SERVER
AQUASCAN_SEND_FPS=${AQUASCAN_SEND_FPS:-5}
AQUASCAN_SEND_QUALITY=${AQUASCAN_SEND_QUALITY:-70}
EOF

echo; echo "===== service file: /etc/systemd/system/aquascan-edge.service ====="
cat hardware/aquascan-edge.service
echo; echo "===== settings: $ENV_FILE ====="
cat "$ENV_FILE.new"
echo
read -r -p "Install and enable this service now (it starts immediately and on every boot)? [y/N] " ans
if [ "${ans,,}" != "y" ]; then rm -f "$ENV_FILE.new"; echo "Nothing changed."; exit 0; fi

mv "$ENV_FILE.new" "$ENV_FILE"
sudo cp hardware/aquascan-edge.service /etc/systemd/system/aquascan-edge.service
sudo systemctl daemon-reload
sudo systemctl enable --now aquascan-edge
sleep 8
systemctl --no-pager --lines=8 status aquascan-edge || true
echo
echo "Installed. Stop: sudo systemctl stop aquascan-edge | Disable: sudo systemctl disable --now aquascan-edge | Logs: journalctl -u aquascan-edge -f"
