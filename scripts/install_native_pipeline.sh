#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${TARGET_DIR:-/opt/rk3568_camera}"
STATE_DIR="${STATE_DIR:-/var/lib/rk3568-camera}"
BUILD_NATIVE=0
ACTIVATE=0
ROLLBACK=0
NATIVE_BUILD_JOBS="${NATIVE_BUILD_JOBS:-1}"

usage() {
  cat <<'EOF'
Usage: sudo bash scripts/install_native_pipeline.sh [--build] [--activate|--rollback]

  --build     Build the native C++ service against the board Rockchip SDK tree.
  --activate  Stop the old snapshot/Python trigger and start the native candidate.
  --rollback  Stop the native candidate and restore the old snapshot/Python trigger.

Optional environment:
  DOCALIGNER_RKNN_SOURCE=/path/to/docaligner_rk3568_uint8_fp16.rknn
  PPOCR_DET_RKNN_SOURCE=/path/to/ppocrv4_det_480x480_fp16.rknn
  NATIVE_BINARY_SOURCE=/path/to/rk3568_native_pipeline_service
  NATIVE_BUILD_JOBS=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD_NATIVE=1 ;;
    --activate) ACTIVATE=1 ;;
    --rollback) ROLLBACK=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$(id -u)" -eq 0 ]] || { echo "Run as root." >&2; exit 1; }
[[ "$ACTIVATE" -eq 0 || "$ROLLBACK" -eq 0 ]] || {
  echo "Choose either --activate or --rollback." >&2
  exit 2
}
[[ "$NATIVE_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "NATIVE_BUILD_JOBS must be a positive integer." >&2
  exit 2
}

old_units=(
  rk3568-camera-snapshot-bridge.service
  rk3568-camera-report-trigger.service
)
native_units=(
  rk3568-camera-native-structured.service
  rk3568-camera-native-pipeline.service
  rk3568-camera-native-preview.service
)

restore_previous_pipeline() {
  systemctl disable --now "${native_units[@]}" || true
  rm -f /etc/systemd/system/rk3568-camera-ocr-overlay.service.d/native-pipeline.conf
  rmdir /etc/systemd/system/rk3568-camera-ocr-overlay.service.d 2>/dev/null || true
  systemctl daemon-reload
  systemctl enable "${old_units[@]}"
  systemctl restart "${old_units[@]}"
  systemctl restart rk3568-camera-ocr-overlay.service
}

if [[ "$ROLLBACK" -eq 1 ]]; then
  restore_previous_pipeline
  echo "Restored the previous RK3568 camera pipeline."
  exit 0
fi

required_commands=(awk install rsync sha256sum systemctl)
if [[ "$BUILD_NATIVE" -eq 1 ]]; then
  required_commands+=(cmake)
fi
for command in "${required_commands[@]}"; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Missing required command: $command" >&2
    exit 1
  }
done

install -d -m 0755 \
  "$TARGET_DIR" "$TARGET_DIR/bin" "$TARGET_DIR/models" \
  "$TARGET_DIR/native" "$TARGET_DIR/report_parser" "$TARGET_DIR/scripts" \
  "$TARGET_DIR/systemd"
install -d -m 0700 "$STATE_DIR"
rsync -a --delete --exclude=build/ "$SOURCE_DIR/native/" "$TARGET_DIR/native/"
rsync -a --exclude=__pycache__/ --exclude=.pytest_cache/ \
  "$SOURCE_DIR/report_parser/" "$TARGET_DIR/report_parser/"
rsync -a --exclude=__pycache__/ --exclude=.pytest_cache/ \
  "$SOURCE_DIR/scripts/" "$TARGET_DIR/scripts/"
chmod 0755 "$TARGET_DIR/scripts/install_native_pipeline.sh" \
  "$TARGET_DIR/scripts/reset_npu_min_frequency.sh" \
  "$TARGET_DIR/scripts/native_preview_server.py"
install -m 0644 "$SOURCE_DIR/systemd/rk3568-camera-native-structured.service" \
  /etc/systemd/system/rk3568-camera-native-structured.service
install -m 0644 "$SOURCE_DIR/systemd/rk3568-camera-native-pipeline.service" \
  /etc/systemd/system/rk3568-camera-native-pipeline.service
install -m 0644 "$SOURCE_DIR/systemd/rk3568-camera-native-preview.service" \
  /etc/systemd/system/rk3568-camera-native-preview.service
if [[ ! -s "$STATE_DIR/native.env" ]]; then
  install -m 0600 "$SOURCE_DIR/config/native.env.example" "$STATE_DIR/native.env"
fi
legacy_rules_sha256="2449e3b110ea7853224f921e8a9bd31be68c97887c43b881d8f20ac004599e05"
current_rules_sha256=""
if [[ -s "$STATE_DIR/active-field-rules.json" ]]; then
  current_rules_sha256="$(sha256sum "$STATE_DIR/active-field-rules.json" | awk '{print $1}')"
fi
if [[ ! -s "$STATE_DIR/active-field-rules.json" || \
      "$current_rules_sha256" == "$legacy_rules_sha256" ]]; then
  install -m 0600 "$SOURCE_DIR/report_parser/runtime/active_fixed_field_rules.json" \
    "$STATE_DIR/active-field-rules.json"
fi

model_source="${DOCALIGNER_RKNN_SOURCE:-$SOURCE_DIR/artifacts/docaligner_rk3568_uint8_fp16.rknn}"
if [[ -s "$model_source" ]]; then
  install -m 0644 "$model_source" \
    "$TARGET_DIR/models/docaligner_rk3568_uint8_fp16.rknn"
fi
ppocr_det_source="${PPOCR_DET_RKNN_SOURCE:-$SOURCE_DIR/artifacts/ppocrv4_det_480x480_fp16.rknn}"
if [[ -s "$ppocr_det_source" ]]; then
  install -m 0644 "$ppocr_det_source" \
    "$TARGET_DIR/models/ppocrv4_det_480x480_fp16.rknn"
fi
binary_source="${NATIVE_BINARY_SOURCE:-$SOURCE_DIR/artifacts/rk3568_native_pipeline_service}"
if [[ -s "$binary_source" ]]; then
  install -m 0755 "$binary_source" \
    "$TARGET_DIR/bin/rk3568_native_pipeline_service.next"
  mv -f "$TARGET_DIR/bin/rk3568_native_pipeline_service.next" \
    "$TARGET_DIR/bin/rk3568_native_pipeline_service"
fi

if [[ "$BUILD_NATIVE" -eq 1 ]]; then
  cmake -S "$TARGET_DIR/native" -B "$TARGET_DIR/native-build" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$TARGET_DIR/native-build" -- -j"$NATIVE_BUILD_JOBS"
  install -m 0755 "$TARGET_DIR/native-build/rk3568_native_pipeline_service" \
    "$TARGET_DIR/bin/rk3568_native_pipeline_service.next"
  mv -f "$TARGET_DIR/bin/rk3568_native_pipeline_service.next" \
    "$TARGET_DIR/bin/rk3568_native_pipeline_service"
fi

[[ -x "$TARGET_DIR/bin/rk3568_native_pipeline_service" ]] || {
  echo "Native binary missing; use --build or NATIVE_BINARY_SOURCE." >&2
  exit 1
}
[[ -s "$TARGET_DIR/models/docaligner_rk3568_uint8_fp16.rknn" ]] || {
  echo "DocAligner RKNN model missing; set DOCALIGNER_RKNN_SOURCE." >&2
  exit 1
}
[[ -x "$TARGET_DIR/.venv-camera/bin/python" ]] || {
  echo "Existing camera Python environment is missing." >&2
  exit 1
}

set -a
# shellcheck disable=SC1090
source "$STATE_DIR/native.env"
set +a
for path in \
  "$DOCALIGNER_MODEL" "$PPOCR_DET_MODEL" "$PPOCR_REC_MODEL" \
  "$RKNN_RUNTIME_DIR/librknnrt.so" "$RGA_RUNTIME_DIR/librga.so" "$VIDEO_DEVICE"; do
  [[ -e "$path" ]] || { echo "Required native pipeline path missing: $path" >&2; exit 1; }
done

systemctl daemon-reload
if [[ "$ACTIVATE" -eq 1 ]]; then
  install -d -m 0755 /etc/systemd/system/rk3568-camera-ocr-overlay.service.d
  install -m 0644 \
    "$SOURCE_DIR/systemd/rk3568-camera-ocr-overlay.native.conf" \
    /etc/systemd/system/rk3568-camera-ocr-overlay.service.d/native-pipeline.conf
  systemctl daemon-reload
  systemctl disable --now "${old_units[@]}" || true
  if ! systemctl enable "${native_units[@]}" rk3568-camera-ocr-overlay.service || \
     ! systemctl restart rk3568-camera-native-structured.service || \
     ! systemctl restart rk3568-camera-native-preview.service || \
     ! systemctl restart rk3568-camera-native-pipeline.service || \
     ! systemctl restart rk3568-camera-ocr-overlay.service || \
     ! systemctl is-active --quiet rk3568-camera-native-structured.service || \
     ! systemctl is-active --quiet rk3568-camera-native-pipeline.service || \
     ! systemctl is-active --quiet rk3568-camera-native-preview.service || \
     ! systemctl is-active --quiet rk3568-camera-ocr-overlay.service; then
    echo "Native activation failed; restoring the previous pipeline." >&2
    restore_previous_pipeline
    exit 1
  fi
  echo "RK3568 native camera candidate activated."
else
  echo "Native files installed but services were not switched."
fi
