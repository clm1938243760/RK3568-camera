#!/usr/bin/env python3
"""Benchmark the stable PP-OCR HTTP contract with one deidentified JPEG."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def percentile(values, percentile_value):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile_value) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values):
    cleaned = [float(value) for value in values]
    if not cleaned:
        return None
    return {
        "count": len(cleaned),
        "min": round(min(cleaned), 2),
        "mean": round(statistics.mean(cleaned), 2),
        "median": round(statistics.median(cleaned), 2),
        "p95": round(percentile(cleaned, 95), 2),
        "max": round(max(cleaned), 2),
    }


def health_endpoint(ocr_endpoint):
    parsed = urlsplit(ocr_endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def read_json_response(request, timeout):
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("endpoint returned a non-object JSON response")
    return payload


def request_health(endpoint, timeout):
    request = Request(endpoint, headers={"Accept": "application/json"})
    payload = read_json_response(request, timeout)
    return {
        "ok": bool(payload.get("ok")),
        "backend": str(payload.get("backend") or ""),
    }


def request_ocr(endpoint, image_bytes, timeout):
    request = Request(
        endpoint,
        data=image_bytes,
        headers={
            "Content-Type": "image/jpeg",
            "Accept": "application/json",
            "User-Agent": "RK3568-Camera-Benchmark/0.1",
        },
        method="POST",
    )
    started = time.monotonic()
    payload = read_json_response(request, timeout)
    client_ms = (time.monotonic() - started) * 1000.0
    if payload.get("ok") is False:
        raise RuntimeError(str(payload.get("error") or "OCR returned ok=false"))
    server_ms = payload.get("elapsed_ms")
    return {
        "client_ms": client_ms,
        "server_ms": float(server_ms) if server_ms is not None else None,
        "ocr_count": len(payload.get("ocr") or []),
    }


def run_benchmark(endpoint, image_bytes, iterations, warmup, timeout):
    for _index in range(warmup):
        request_ocr(endpoint, image_bytes, timeout)

    samples = []
    for index in range(iterations):
        sample = request_ocr(endpoint, image_bytes, timeout)
        sample["iteration"] = index + 1
        samples.append(sample)

    return {
        "client_elapsed_ms": summarize([item["client_ms"] for item in samples]),
        "server_elapsed_ms": summarize(
            [item["server_ms"] for item in samples if item["server_ms"] is not None]
        ),
        "ocr_count": summarize([item["ocr_count"] for item in samples]),
        "samples": [
            {
                "iteration": item["iteration"],
                "client_ms": round(item["client_ms"], 2),
                "server_ms": (
                    round(item["server_ms"], 2) if item["server_ms"] is not None else None
                ),
                "ocr_count": item["ocr_count"],
            }
            for item in samples
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RK3568/RK3588 PP-OCR HTTP latency")
    parser.add_argument("--image", type=Path, required=True, help="deidentified JPEG fixture")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5002/ocr")
    parser.add_argument("--board-label", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("iterations must be positive and warmup cannot be negative")
    image_bytes = args.image.read_bytes()
    if not image_bytes.startswith(b"\xff\xd8"):
        raise SystemExit("--image must be a JPEG")

    result = {
        "schema_version": 1,
        "kind": "fixed_fixture_ocr_http",
        "board_label": args.board_label,
        "created_at": time.time(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_bytes": len(image_bytes),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "health": request_health(health_endpoint(args.endpoint), args.timeout),
        "metrics": run_benchmark(
            args.endpoint,
            image_bytes,
            args.iterations,
            args.warmup,
            args.timeout,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
