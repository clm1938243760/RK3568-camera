#!/usr/bin/env python3
"""Collect PHI-free timing records from camera status and result JSON files."""

from __future__ import print_function

import argparse
import json
import statistics
import time
from pathlib import Path


TIMING_KEYS = (
    "paper_to_result_ms",
    "stability_ms",
    "quality_ms",
    "crop_ms",
    "transform_ms",
    "ocr_ms",
    "ocr_detection_ms",
    "ocr_crop_ms",
    "ocr_recognition_preprocess_ms",
    "ocr_recognition_inference_ms",
    "ocr_recognition_postprocess_ms",
    "structured_ms",
    "post_stable_ms",
    "paper_to_ocr_ms",
    "total_ms",
)


def read_json(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def summarize(values):
    if not values:
        return None
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


class TimingCollector(object):
    def __init__(self, board_label):
        self.board_label = board_label
        self.paper_started_at = None
        self.capture_started = {}
        self.seen_results = set()
        self.records = []

    def observe_status(self, payload):
        updated_at_ms = payload.get("updated_at_ms")
        now = float(
            payload.get("updated_at")
            or (
                float(updated_at_ms) / 1000.0
                if isinstance(updated_at_ms, (int, float))
                else time.time()
            )
        )
        detected = bool(payload.get("paper_detected"))
        if detected and self.paper_started_at is None:
            self.paper_started_at = now
        capture_id = str(payload.get("capture_id") or "")
        if capture_id and capture_id not in self.capture_started:
            self.capture_started[capture_id] = self.paper_started_at or now
        if not detected:
            self.paper_started_at = None

    def observe_result(self, payload):
        capture_id = str(payload.get("capture_id") or "")
        if not capture_id or capture_id in self.seen_results:
            return None
        timings = payload.get("timings") or {}
        if not isinstance(timings, dict):
            return None
        observed_at = time.time()
        paper_started = self.capture_started.pop(capture_id, None)
        record = {
            "board_label": self.board_label,
            "capture_id": capture_id,
            "observed_at": observed_at,
            "result_status": str(payload.get("status") or ""),
            "paper_to_result_ms": (
                round(max(0.0, observed_at - paper_started) * 1000.0, 2)
                if paper_started is not None
                else None
            ),
            "field_count": len(payload.get("fields") or {}),
            "ocr_item_count": int(((payload.get("source") or {}).get("ocr_item_count") or 0)),
        }
        for key, value in timings.items():
            if isinstance(value, (int, float)):
                record[str(key)] = round(float(value), 2)
        self.seen_results.add(capture_id)
        self.records.append(record)
        return record

    def output(self):
        summary = {}
        for key in TIMING_KEYS:
            values = [
                float(record[key])
                for record in self.records
                if isinstance(record.get(key), (int, float))
            ]
            if values:
                summary[key] = summarize(values)
        return {
            "schema_version": 1,
            "kind": "live_camera_end_to_end",
            "board_label": self.board_label,
            "created_at": time.time(),
            "sample_count": len(self.records),
            "summary": summary,
            "records": self.records,
        }


def write_output(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect live paper-to-result timing samples")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("/run/rk3568-camera/camera-trigger.json"),
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=Path("/run/rk3568-camera/structured-result.json"),
    )
    parser.add_argument("--board-label", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 1 or args.poll_seconds <= 0:
        raise SystemExit("samples and poll-seconds must be positive")
    collector = TimingCollector(args.board_label)
    last_status_mtime = None
    last_result_mtime = None
    try:
        while len(collector.records) < args.samples:
            try:
                status_mtime = args.status_file.stat().st_mtime_ns
            except OSError:
                status_mtime = None
            if status_mtime is not None and status_mtime != last_status_mtime:
                status = read_json(args.status_file)
                if status is not None:
                    collector.observe_status(status)
                last_status_mtime = status_mtime

            try:
                result_mtime = args.result_file.stat().st_mtime_ns
            except OSError:
                result_mtime = None
            if result_mtime is not None and result_mtime != last_result_mtime:
                result = read_json(args.result_file)
                if result is not None:
                    record = collector.observe_result(result)
                    if record is not None:
                        print(
                            "sample=%d capture=%s total_ms=%s paper_to_result_ms=%s"
                            % (
                                len(collector.records),
                                record["capture_id"][:8],
                                record.get("total_ms"),
                                record.get("paper_to_result_ms"),
                            ),
                            flush=True,
                        )
                        write_output(args.output, collector.output())
                last_result_mtime = result_mtime
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        pass
    write_output(args.output, collector.output())
    print(json.dumps(collector.output()["summary"], sort_keys=True))
    return 0 if collector.records else 1


if __name__ == "__main__":
    raise SystemExit(main())
