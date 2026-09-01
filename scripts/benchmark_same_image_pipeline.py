#!/usr/bin/env python3
"""Benchmark one private image through the production full-text OCR path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paper detection, rectification and configured-region full-text OCR"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ocr-endpoint", default="http://127.0.0.1:5002/ocr")
    parser.add_argument("--region-top", type=float, default=0.13)
    parser.add_argument("--region-bottom", type=float, default=0.60)
    parser.add_argument("--tile-max-aspect", type=float, default=2.0)
    parser.add_argument("--tile-overlap-ratio", type=float, default=0.15)
    parser.add_argument("--tile-max-count", type=int, default=2)
    parser.add_argument("--detector-warmup", type=int, default=3)
    parser.add_argument("--detector-iterations", type=int, default=20)
    parser.add_argument("--rectification-iterations", type=int, default=5)
    parser.add_argument("--crop-iterations", type=int, default=5)
    parser.add_argument("--ocr-warmup", type=int, default=1)
    parser.add_argument("--ocr-iterations", type=int, default=3)
    parser.add_argument("--full-iterations", type=int, default=3)
    return parser.parse_args()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("timing sample must not be empty")
    index = max(
        0,
        min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)),
    )
    return float(ordered[index])


def timing_summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except OSError:
        return ""


def timed_imports() -> Dict[str, float]:
    timings: Dict[str, float] = {}
    started = time.perf_counter()
    import numpy  # noqa: F401

    timings["numpy_ms"] = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    import cv2  # noqa: F401

    timings["opencv_ms"] = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    import onnxruntime  # noqa: F401

    timings["onnxruntime_ms"] = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    from PIL import Image  # noqa: F401

    timings["pillow_ms"] = (time.perf_counter() - started) * 1000.0
    return {key: round(value, 3) for key, value in timings.items()}


def main() -> int:
    args = parse_args()
    for name in (
        "detector_warmup",
        "detector_iterations",
        "rectification_iterations",
        "crop_iterations",
        "ocr_warmup",
        "ocr_iterations",
        "full_iterations",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError("%s must be at least one" % name.replace("_", "-"))
    if not 0.0 <= args.region_top < args.region_bottom <= 1.0:
        raise ValueError("configured recognition region must be inside the document")

    import_timings = timed_imports()
    import cv2
    import numpy
    import onnxruntime
    from PIL import Image

    from rk3588_report_parser.capture_orientation import prepare_document_jpeg
    from rk3588_report_parser.capture_region import (
        DocumentRecognitionRegion,
        crop_document_jpeg,
        remap_extraction_to_full_document,
    )
    from rk3588_report_parser.capture_text import FullTextExtractor, TextRefinementSettings
    from rk3588_report_parser.clients import LocalPpOcrClient
    from rk3588_report_parser.paper_detector import create_docaligner_detector
    from rk3588_report_parser.settings import OcrSettings

    image_bytes = args.image.read_bytes()
    model_bytes = args.model.read_bytes()
    with Image.open(args.image) as source:
        image_size = list(source.size)

    detector = create_docaligner_detector(args.model, 0.5, "onnxruntime")
    for _ in range(args.detector_warmup):
        detector.detect_jpeg(image_bytes)

    inference_ms: List[float] = []
    detector_total_ms: List[float] = []
    detection = None
    for _ in range(args.detector_iterations):
        started = time.perf_counter()
        detection = detector.detect_jpeg(image_bytes)
        detector_total_ms.append((time.perf_counter() - started) * 1000.0)
        inference_ms.append(float(detection.inference_ms))
    if detection is None or not detection.detected:
        raise RuntimeError("DocAligner did not detect the benchmark document")

    rectification_ms: List[float] = []
    rectified = b""
    for _ in range(args.rectification_iterations):
        started = time.perf_counter()
        rectified = prepare_document_jpeg(
            image_bytes,
            detection,
            degrees_counterclockwise=90,
            target_long_side=3200,
        )
        rectification_ms.append((time.perf_counter() - started) * 1000.0)

    from io import BytesIO

    with Image.open(BytesIO(rectified)) as prepared:
        rectified_size = list(prepared.size)

    recognition_region = DocumentRecognitionRegion(
        crop_top=args.region_top,
        crop_bottom=args.region_bottom,
        accept_top=args.region_top,
        accept_bottom=args.region_bottom,
    )
    crop_ms: List[float] = []
    cropped = None
    for _ in range(args.crop_iterations):
        started = time.perf_counter()
        cropped = crop_document_jpeg(rectified, recognition_region)
        crop_ms.append((time.perf_counter() - started) * 1000.0)
    if cropped is None:
        raise RuntimeError("configured recognition region was not cropped")
    with Image.open(BytesIO(cropped.image_bytes)) as region_image:
        region_size = list(region_image.size)

    extractor = FullTextExtractor(
        LocalPpOcrClient(),
        OcrSettings(args.ocr_endpoint, 30.0),
        TextRefinementSettings(
            max_regions=0,
            max_duration_seconds=10.0,
            primary_tile_max_aspect=args.tile_max_aspect,
            primary_tile_overlap_ratio=args.tile_overlap_ratio,
            primary_tile_max_count=args.tile_max_count,
        ),
    )

    def recognize_region(region_bytes: bytes, mapping: Any) -> Any:
        result = extractor.extract_refined(region_bytes)
        return remap_extraction_to_full_document(
            result,
            mapping,
            low_confidence=extractor.refinement.low_confidence,
            low_mean_confidence=extractor.refinement.low_mean_confidence,
        )

    for _ in range(args.ocr_warmup):
        recognize_region(cropped.image_bytes, cropped)

    ocr_ms: List[float] = []
    extraction = None
    for _ in range(args.ocr_iterations):
        started = time.perf_counter()
        extraction = recognize_region(cropped.image_bytes, cropped)
        ocr_ms.append((time.perf_counter() - started) * 1000.0)
    if extraction is None or extraction.document is None:
        raise RuntimeError("configured-region full-text OCR did not produce a document")

    full_total_ms: List[float] = []
    full_detection_ms: List[float] = []
    full_rectification_ms: List[float] = []
    full_crop_ms: List[float] = []
    full_ocr_ms: List[float] = []
    for _ in range(args.full_iterations):
        full_started = time.perf_counter()
        started = time.perf_counter()
        current_detection = detector.detect_jpeg(image_bytes)
        full_detection_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        current_rectified = prepare_document_jpeg(
            image_bytes,
            current_detection,
            degrees_counterclockwise=90,
            target_long_side=3200,
        )
        full_rectification_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        current_crop = crop_document_jpeg(current_rectified, recognition_region)
        full_crop_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        current_extraction = recognize_region(current_crop.image_bytes, current_crop)
        full_ocr_ms.append((time.perf_counter() - started) * 1000.0)
        full_total_ms.append((time.perf_counter() - full_started) * 1000.0)
        if current_extraction.status != extraction.status:
            raise RuntimeError("full pipeline status changed between iterations")

    document = extraction.document
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "benchmark_mode": "configured_region_full_text",
        "board": os.uname().nodename,
        "runtime": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
            "onnxruntime": onnxruntime.__version__,
            "cpu_count": os.cpu_count(),
            "cpu_governor": read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            ),
            "cpu_current_khz": read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            ),
            "cpu_max_khz": read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
            ),
        },
        "artifacts": {
            "image_sha256": sha256_bytes(image_bytes),
            "image_bytes": len(image_bytes),
            "image_size": image_size,
            "model_sha256": sha256_bytes(model_bytes),
            "rectified_size": rectified_size,
            "recognition_region": recognition_region.to_dict(),
            "recognition_region_size": region_size,
            "tile_max_aspect": args.tile_max_aspect,
            "tile_max_count": args.tile_max_count,
        },
        "cold_imports": import_timings,
        "detector": {
            "backend": detector.backend_name,
            "model_load_ms": round(detector.model_load_ms, 3),
            "confidence": round(detection.confidence, 6),
            "inference": timing_summary(inference_ms),
            "decode_resize_inference_post": timing_summary(detector_total_ms),
        },
        "rectification": timing_summary(rectification_ms),
        "recognition_region_crop": timing_summary(crop_ms),
        "full_text_ocr": {
            "status": extraction.status,
            "reasons": list(extraction.reasons),
            "item_count": len(document.spans),
            "line_count": len(document.lines),
            "mean_confidence": round(document.mean_confidence, 6),
            "recognition_sources": sorted(
                set(span.recognition_source for span in document.spans)
            ),
            "ocr_call_count": int(round(extraction.timings.get("primary_tile_count", 1.0))),
            "total": timing_summary(ocr_ms),
        },
        "full_warm_pipeline": {
            "iterations": args.full_iterations,
            "detection": timing_summary(full_detection_ms),
            "rectification": timing_summary(full_rectification_ms),
            "recognition_region_crop": timing_summary(full_crop_ms),
            "full_text_ocr": timing_summary(full_ocr_ms),
            "total": timing_summary(full_total_ms),
        },
        "privacy": {"ocr_text_emitted": False, "image_retained_by_script": False},
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": "%s:%s" % (type(exc).__name__, exc)},
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
