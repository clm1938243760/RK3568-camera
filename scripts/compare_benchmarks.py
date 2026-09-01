#!/usr/bin/env python3
"""Compare fixed-fixture or live timing summaries from RK3588 and RK3568."""

from __future__ import print_function

import argparse
import json
from pathlib import Path


def nested(payload, path):
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def metric_rows(rk3588, rk3568):
    candidates = (
        ("OCR HTTP client median", "metrics.client_elapsed_ms.median"),
        ("OCR server median", "metrics.server_elapsed_ms.median"),
        ("Paper to result median", "summary.paper_to_result_ms.median"),
        ("Pipeline total median", "summary.total_ms.median"),
        ("Stability median", "summary.stability_ms.median"),
        ("Final quality median", "summary.quality_ms.median"),
        ("RGA crop median", "summary.crop_ms.median"),
        ("Transform median", "summary.transform_ms.median"),
        ("OCR median", "summary.ocr_ms.median"),
        ("Structured fields median", "summary.structured_ms.median"),
    )
    rows = []
    for label, path in candidates:
        left = nested(rk3588, path)
        right = nested(rk3568, path)
        if left is None or right is None:
            continue
        left = float(left)
        right = float(right)
        rows.append(
            {
                "metric": label,
                "rk3588_ms": round(left, 2),
                "rk3568_ms": round(right, 2),
                "delta_ms": round(right - left, 2),
                "rk3568_over_rk3588": round(right / left, 3) if left else None,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RK3588 and RK3568 camera timings")
    parser.add_argument("--rk3588", type=Path, required=True)
    parser.add_argument("--rk3568", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rk3588 = json.loads(args.rk3588.read_text(encoding="utf-8"))
    rk3568 = json.loads(args.rk3568.read_text(encoding="utf-8"))
    left_hash = rk3588.get("image_sha256")
    right_hash = rk3568.get("image_sha256")
    if left_hash and right_hash and left_hash != right_hash:
        raise SystemExit("fixture SHA-256 differs; comparison would not be fair")
    rows = metric_rows(rk3588, rk3568)
    if not rows:
        raise SystemExit("no comparable timing metrics found")

    result = {
        "schema_version": 1,
        "rk3588_source": str(args.rk3588),
        "rk3568_source": str(args.rk3568),
        "image_sha256": left_hash or right_hash,
        "metrics": rows,
    }
    print("| Metric | RK3588 ms | RK3568 ms | Delta ms | RK3568/RK3588 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {metric} | {rk3588_ms:.2f} | {rk3568_ms:.2f} | {delta_ms:.2f} | {ratio} |".format(
                metric=row["metric"],
                rk3588_ms=row["rk3588_ms"],
                rk3568_ms=row["rk3568_ms"],
                delta_ms=row["delta_ms"],
                ratio=(
                    "%.3f" % row["rk3568_over_rk3588"]
                    if row["rk3568_over_rk3588"] is not None
                    else "-"
                ),
            )
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
