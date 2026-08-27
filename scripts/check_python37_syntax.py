#!/usr/bin/env python3
"""Check the deployed RK3568 camera subset against Python 3.7 grammar."""

from __future__ import print_function

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FILES = (
    "camera_ocr_overlay.py",
    "scripts/rk3568_snapshot_bridge.py",
    "scripts/benchmark_ocr_http.py",
    "scripts/compare_benchmarks.py",
    "scripts/collect_e2e_timings.py",
    "report_parser/scripts/camera_paper_trigger.py",
    "report_parser/src/rk3588_report_parser/capture_text_runtime.py",
    "report_parser/src/rk3588_report_parser/capture_orientation.py",
    "report_parser/src/rk3588_report_parser/capture_region.py",
    "report_parser/src/rk3588_report_parser/capture_text.py",
    "report_parser/src/rk3588_report_parser/clients.py",
    "report_parser/src/rk3588_report_parser/frame_quality.py",
    "report_parser/src/rk3588_report_parser/frame_source.py",
    "report_parser/src/rk3588_report_parser/identifier_candidates.py",
    "report_parser/src/rk3588_report_parser/identifier_models.py",
    "report_parser/src/rk3588_report_parser/identifier_rules.py",
    "report_parser/src/rk3588_report_parser/models.py",
    "report_parser/src/rk3588_report_parser/paper_detector.py",
    "report_parser/src/rk3588_report_parser/paper_trigger.py",
    "report_parser/src/rk3588_report_parser/preprocessing.py",
    "report_parser/src/rk3588_report_parser/settings.py",
    "report_parser/src/rk3588_report_parser/spans.py",
    "report_parser/src/rk3588_report_parser/typing_compat.py",
)


def parse_python37(path):
    source = path.read_text(encoding="utf-8")
    if sys.version_info >= (3, 8):
        ast.parse(source, filename=str(path), feature_version=(3, 7))
    else:
        compile(source, str(path), "exec")


def main():
    failures = []
    for relative in PRODUCTION_FILES:
        path = ROOT / relative
        try:
            parse_python37(path)
        except (OSError, SyntaxError) as exc:
            failures.append("%s: %s" % (relative, exc))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("python37 production syntax: %d files OK" % len(PRODUCTION_FILES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
