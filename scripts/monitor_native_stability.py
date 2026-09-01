#!/usr/bin/env python3
"""Monitor the native camera service without retaining OCR or patient data."""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _service_pid(service):
    value = subprocess.check_output(
        ["systemctl", "show", service, "-p", "MainPID", "--value"],
        universal_newlines=True,
    ).strip()
    return int(value or "0")


def _process_metrics(pid):
    metrics = {"rss_kib": 0, "threads": 0, "fd_count": 0}
    for line in Path("/proc/%d/status" % pid).read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            metrics["rss_kib"] = int(line.split()[1])
        elif line.startswith("Threads:"):
            metrics["threads"] = int(line.split()[1])
    metrics["fd_count"] = len(list(Path("/proc/%d/fd" % pid).iterdir()))
    return metrics


def summarize(samples, failures, started_at, finished_at):
    pids = sorted(set(sample["pid"] for sample in samples))
    maximum_status_age_ms = (
        max(sample["status_age_ms"] for sample in samples) if samples else None
    )
    fd_growth = samples[-1]["fd_count"] - samples[0]["fd_count"] if samples else 0
    stages = {}
    for sample in samples:
        stage = sample.get("stage") or "unknown"
        stages[stage] = stages.get(stage, 0) + 1
    return {
        "schema_version": 1,
        "kind": "native_idle_stability",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(finished_at - started_at, 3),
        "sample_count": len(samples),
        "ok": (
            bool(samples)
            and not failures
            and len(pids) == 1
            and fd_growth <= 2
            and maximum_status_age_ms is not None
            and maximum_status_age_ms <= 2000.0
        ),
        "pid_count": len(pids),
        "rss_kib": {
            "min": min(sample["rss_kib"] for sample in samples) if samples else 0,
            "max": max(sample["rss_kib"] for sample in samples) if samples else 0,
        },
        "fd_count": {
            "min": min(sample["fd_count"] for sample in samples) if samples else 0,
            "max": max(sample["fd_count"] for sample in samples) if samples else 0,
            "growth": fd_growth,
        },
        "threads": {
            "min": min(sample["threads"] for sample in samples) if samples else 0,
            "max": max(sample["threads"] for sample in samples) if samples else 0,
        },
        "maximum_status_age_ms": (
            round(maximum_status_age_ms, 2)
            if maximum_status_age_ms is not None
            else None
        ),
        "stages": stages,
        "failures": failures,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor the RK3568 native camera service")
    parser.add_argument(
        "--service", default="rk3568-camera-native-pipeline.service"
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("/run/rk3568-camera/camera-trigger.json"),
    )
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0 or args.interval <= 0:
        raise SystemExit("duration and interval must be positive")
    started_at = time.time()
    deadline = time.monotonic() + args.duration
    samples = []
    failures = []
    while time.monotonic() < deadline:
        observed_at = time.time()
        try:
            pid = _service_pid(args.service)
            if pid <= 0:
                raise RuntimeError("service has no main process")
            process = _process_metrics(pid)
            status = json.loads(args.status_file.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise ValueError("status root is not an object")
            samples.append(
                {
                    "pid": pid,
                    "rss_kib": process["rss_kib"],
                    "fd_count": process["fd_count"],
                    "threads": process["threads"],
                    "stage": str(status.get("stage") or "unknown")[:48],
                    "status_age_ms": max(
                        0.0, (observed_at - args.status_file.stat().st_mtime) * 1000.0
                    ),
                }
            )
        except Exception as error:
            failures.append(type(error).__name__)
        time.sleep(min(args.interval, max(0.0, deadline - time.monotonic())))
    result = summarize(samples, failures, started_at, time.time())
    _atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
