#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${TARGET_DIR:-/opt/rk3568_camera}"
STATE_DIR="${STATE_DIR:-/var/lib/rk3568-camera}"
BOOTSTRAP_PYTHON=0
ACTIVATE=0

usage() {
  cat <<'EOF'
Usage: sudo bash install.sh [--bootstrap-python] [--activate]

  --bootstrap-python  Create the Python 3.7 system-site virtual environment.
  --activate          Enable and restart only the three rk3568-camera services.

The existing frame, PP-OCR, gateway, and USB gadget services are not modified.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap-python)
      BOOTSTRAP_PYTHON=1
      ;;
    --activate)
      ACTIVATE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

[[ "$(id -u)" -eq 0 ]] || { echo "Run as root." >&2; exit 1; }
for command in install rsync python3 systemctl; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Missing required command: $command" >&2
    exit 1
  }
done
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 7) else 1)' || {
  echo "Python 3.7 or newer is required." >&2
  exit 1
}

install -d -m 0755 "$TARGET_DIR" "$TARGET_DIR/scripts" "$TARGET_DIR/systemd"
install -d -m 0700 "$STATE_DIR"
install -m 0644 "$SOURCE_DIR/camera_ocr_overlay.py" "$TARGET_DIR/camera_ocr_overlay.py"
install -m 0644 "$SOURCE_DIR/VERSION" "$SOURCE_DIR/README.md" "$TARGET_DIR/"
rsync -a --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude='.venv*/' "$SOURCE_DIR/scripts/" "$TARGET_DIR/scripts/"
rsync -a --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude='.venv*/' "$SOURCE_DIR/systemd/" "$TARGET_DIR/systemd/"
rsync -a --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude='.venv*/' "$SOURCE_DIR/report_parser/" "$TARGET_DIR/report_parser/"

chmod 0755 "$TARGET_DIR/scripts/"*.py "$TARGET_DIR/scripts/"*.sh

if [[ ! -s "$STATE_DIR/camera.env" ]]; then
  install -m 0600 "$SOURCE_DIR/config/camera.env.example" "$STATE_DIR/camera.env"
fi
if [[ ! -s "$STATE_DIR/active_identifier_rules.json" ]]; then
  install -m 0600 "$SOURCE_DIR/report_parser/runtime/active_identifier_rules.json" "$STATE_DIR/active_identifier_rules.json"
fi

if [[ ! -x "$TARGET_DIR/.venv-camera/bin/python" ]]; then
  [[ "$BOOTSTRAP_PYTHON" -eq 1 ]] || {
    echo "$TARGET_DIR/.venv-camera is missing; rerun with --bootstrap-python" >&2
    exit 1
  }
  python3 -m venv --system-site-packages "$TARGET_DIR/.venv-camera"
fi

"$TARGET_DIR/.venv-camera/bin/python" -c 'import cv2, numpy; from PIL import Image; print("RK3568 camera Python dependencies OK")'

units=(
  rk3568-camera-snapshot-bridge.service
  rk3568-camera-report-trigger.service
  rk3568-camera-ocr-overlay.service
)
for unit in "${units[@]}"; do
  install -m 0644 "$SOURCE_DIR/systemd/$unit" "/etc/systemd/system/$unit"
done
systemctl daemon-reload

if [[ "$ACTIVATE" -eq 1 ]]; then
  systemctl enable "${units[@]}"
  for unit in "${units[@]}"; do
    systemctl restart "$unit"
  done
  echo "RK3568 camera recognition services activated."
else
  echo "Files and units installed but not enabled or started."
  echo "Review $STATE_DIR/camera.env, then run: sudo bash $SOURCE_DIR/install.sh --activate"
fi

echo "Existing rk3568-patient-frame, rk3568-ppocr, gateway and USB services were not modified."
