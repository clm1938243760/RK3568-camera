#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${OUTPUT:-}"
if [[ -n "$OUTPUT" ]]; then
  mkdir -p "$(dirname "$OUTPUT")"
  exec > >(tee "$OUTPUT") 2>&1
fi

section() {
  printf '\n[%s]\n' "$1"
}

run_optional() {
  "$@" 2>&1 || true
}

section "identity"
date -Is
run_optional uname -a
run_optional sh -c 'cat /proc/device-tree/model 2>/dev/null; printf "\n"'
run_optional python3 --version
run_optional sh -c 'grep -E "^(MemTotal|MemAvailable):" /proc/meminfo'

section "npu"
run_optional sh -c 'ls -l /dev/rknpu* 2>/dev/null'
run_optional sh -c 'find -L /sys/class/devfreq -maxdepth 1 -type d -name "*npu*" -print'
run_optional sh -c 'for node in /sys/class/devfreq/*npu*; do [ -d "$node" ] || continue; printf "%s " "$node"; cat "$node/cur_freq" 2>/dev/null || true; done'
run_optional sh -c 'ldconfig -p 2>/dev/null | grep -E "rknn|rga"'

section "video"
run_optional sh -c 'for name in /sys/class/video4linux/video*/name; do [ -f "$name" ] || continue; printf "%s: " "$(basename "$(dirname "$name")")"; cat "$name"; done'
if command -v v4l2-ctl >/dev/null 2>&1; then
  run_optional v4l2-ctl --list-devices
  if [[ -e /dev/video9 ]]; then
    run_optional v4l2-ctl -d /dev/video9 --get-fmt-video
    run_optional v4l2-ctl -d /dev/video9 --list-formats-ext
  fi
fi

section "services"
run_optional systemctl is-active rk3568-patient-frame.service
run_optional systemctl is-active rk3568-ppocr.service
run_optional curl -fsS --max-time 5 http://127.0.0.1:8090/health
printf '\n'
run_optional curl -fsS --max-time 5 http://127.0.0.1:5002/health
printf '\n'

section "python modules"
run_optional python3 -c 'import sys; print("python=%s" % sys.version.replace("\n", " "))'
run_optional python3 -c 'import cv2; print("opencv=%s" % cv2.__version__)'
run_optional python3 -c 'import numpy; print("numpy=%s" % numpy.__version__)'
run_optional python3 -c 'import PIL; print("pillow=%s" % PIL.__version__)'

section "runtime files"
run_optional sh -c 'find /userdata /opt -maxdepth 6 -type f \( -name "ppocrv4_det.rknn" -o -name "ppocrv4_rec.rknn" -o -name "librknnrt.so" -o -name "rknn_ppocr_system_worker" \) -print 2>/dev/null'

section "result"
echo "Preflight is read-only. No camera, NPU, service, or USB state was changed."
