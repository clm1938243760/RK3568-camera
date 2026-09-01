#!/bin/sh
set -eu

frequency_path="${NPU_MIN_FREQUENCY_PATH:-/sys/class/devfreq/fde40000.npu/min_freq}"
idle_frequency_hz="${NPU_IDLE_MIN_FREQUENCY_HZ:-200000000}"

if [ -w "$frequency_path" ]; then
  printf '%s\n' "$idle_frequency_hz" >"$frequency_path"
fi
