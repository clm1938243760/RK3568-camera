#!/usr/bin/env python3
"""Event-driven bridge from native RKNN OCR output to RK3588 field rules."""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rk3588_report_parser.native_patient_resolver import (
    CameraPatientResolver,
)
from rk3588_report_parser.spans import build_spans


def _atomic_write(path: Path, payload: Dict[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        ".%s.%d.%d.tmp" % (path.name, os.getpid(), int(time.monotonic() * 1000000))
    )
    descriptor = os.open(
        str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, mode)
        else:
            os.chmod(str(temporary), mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _normalize_schema(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("schema"), dict):
        payload = payload["schema"]
    fields = payload.get("fields") if isinstance(payload, dict) else None
    if not isinstance(fields, list) or not fields:
        raise ValueError("field schema must contain configured fields")
    normalized = []
    for raw in fields:
        if not isinstance(raw, dict):
            continue
        field_key = str(raw.get("field_key") or "").strip()
        if not field_key:
            continue
        aliases = raw.get("label_aliases")
        if not isinstance(aliases, list):
            aliases = []
        label = str(raw.get("label") or raw.get("prompt") or "").strip()
        if label and label not in aliases:
            aliases = [label] + list(aliases)
        fixed_length = int(raw.get("fixed_length") or 0)
        maximum_distance = float(raw.get("max_distance", 0.25))
        if maximum_distance > 1.0:
            maximum_distance /= 1000.0
        minimum_score = float(raw.get("min_ocr_score", 0.65))
        if not 0.0 <= minimum_score <= 1.0 or not 0.0 <= maximum_distance <= 1.0:
            raise ValueError("field score or distance is out of range")
        relations = raw.get("relations")
        if not relations:
            position = str(raw.get("position") or "right_then_below")
            positions = {
                "right": ["same_text", "same_line_right"],
                "below": ["same_text", "next_line_same_column"],
                "right_then_below": ["same_text", "same_line_right", "next_line_same_column"],
            }
            if position not in positions:
                raise ValueError("invalid field position")
            relations = positions[position]
        normalized.append(
            {
                "field_key": field_key,
                "target": str(raw.get("target") or field_key),
                "enabled": bool(raw.get("enabled", True)),
                "required": bool(raw.get("required", False)),
                "label_aliases": aliases,
                "char_type": str(raw.get("char_type") or "any"),
                "lengths": list(raw.get("lengths") or ([fixed_length] if fixed_length else [])),
                "min_length": int(raw.get("min_length") or 0),
                "max_length": int(raw.get("max_length") or 10000),
                "min_ocr_score": minimum_score,
                "max_distance": maximum_distance,
                "relations": list(relations),
                "regex": str(raw.get("regex") or ""),
                "roi": raw.get("roi"),
                "match_mode": str(raw.get("match_mode") or ""),
                "join_mode": str(raw.get("join_mode") or "single"),
                "join_separator": str(raw.get("join_separator") or ""),
            }
        )
    if not normalized or not any(field["enabled"] for field in normalized):
        raise ValueError("field schema has no enabled fields")
    return normalized


def _load_schema(path: Path, endpoint: str = "") -> List[Dict[str, Any]]:
    if endpoint:
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(endpoint, timeout=1.5, context=context) as response:
                return _normalize_schema(json.loads(response.read().decode("utf-8")))
        except Exception:
            pass
    return _normalize_schema(_load_json(path))


def _group_lines(spans: Iterable[Any]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Any]] = {}
    for span in spans:
        grouped.setdefault(int(span.line_id), []).append(span)
    lines = []
    for line_id in sorted(grouped):
        ordered = sorted(grouped[line_id], key=lambda item: (item.box[0], item.source_index))
        lines.append(
            {
                "line_id": line_id,
                "text": " ".join(item.text for item in ordered),
                "span_ids": [item.id for item in ordered],
            }
        )
    return lines


def _public_fields(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fields: Dict[str, Dict[str, Any]] = {}
    for key, value in evidence.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        raw_span_ids = value.get("span_ids")
        span_ids = [
            int(item)
            for item in (raw_span_ids if isinstance(raw_span_ids, list) else [])
            if isinstance(item, int) and not isinstance(item, bool) and item > 0
        ]
        if not span_ids:
            continue
        fields[key] = {
            "value": str(value.get("value") or ""),
            "probability": round(max(0.0, min(1.0, float(value.get("score") or 0.0))), 4),
            "source_span_ids": span_ids[:32],
            "matched_prompt": str(value.get("label") or "")[:80],
            "relation": str(value.get("relation") or "")[:48],
        }
    return fields


def process_once(
    input_path: Path,
    schema_path: Path,
    full_text_path: Path,
    structured_path: Path,
    status_path: Path,
    schema_endpoint: str = "",
) -> Dict[str, Any]:
    started = time.monotonic()
    raw = _load_json(input_path)
    image_size_raw = raw.get("image_size") or [0, 0]
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    spans = tuple(build_spans({"ocr": raw.get("ocr", [])}, image_size))
    lines = _group_lines(spans)
    mean_confidence = (
        sum(float(span.score) for span in spans) / len(spans) if spans else 0.0
    )
    document = {
        "schema_version": 2,
        "image_size": list(image_size),
        "full_text": "\n".join(str(line["text"]) for line in lines),
        "lines": lines,
        "blocks": [span.to_dict() for span in spans],
        "line_count": len(lines),
        "item_count": len(spans),
        "mean_confidence": round(mean_confidence, 4),
    }
    capture_id = str(raw.get("capture_id") or "")
    full_text_payload = {
        "status": "accepted" if spans else "rejected",
        "capture_id": capture_id,
        "created_at": time.time(),
        "source": raw.get("source") if isinstance(raw.get("source"), dict) else {},
        "timings": raw.get("timings") if isinstance(raw.get("timings"), dict) else {},
        "document": document,
    }
    _atomic_write(full_text_path, full_text_payload, 0o600)

    resolved = CameraPatientResolver().resolve(
        full_text_payload, _load_schema(schema_path, schema_endpoint)
    )
    fields = _public_fields(resolved["evidence"])
    review_fields = list(
        dict.fromkeys(
            list(resolved["missing_fields"]) + list(resolved["conflict_fields"])
        )
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    raw_timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}
    ocr_ms = float(raw_timings.get("ocr_ms") or 0.0)
    timing_keys = (
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
        "post_stable_ms",
        "paper_to_ocr_ms",
    )
    timings = {
        key: round(float(raw_timings[key]), 2)
        for key in timing_keys
        if isinstance(raw_timings.get(key), (int, float))
    }
    timings["uie_ms"] = round(elapsed_ms, 2)
    timings["structured_ms"] = round(elapsed_ms, 2)
    timings["total_ms"] = round(
        float(raw_timings.get("paper_to_ocr_ms") or ocr_ms) + elapsed_ms,
        2,
    )
    structured_payload = {
        "capture_id": capture_id,
        "created_at": time.time(),
        "status": resolved["status"],
        "patient_response": resolved["response"],
        "fields": fields,
        "review_fields": review_fields,
        "missing_fields": resolved["missing_fields"],
        "conflict_fields": resolved["conflict_fields"],
        "source": {
            "ocr_item_count": len(spans),
            "mean_confidence": round(mean_confidence, 4),
        },
        "timings": timings,
    }
    _atomic_write(structured_path, structured_payload, 0o600)
    public_status = {
        "ok": True,
        "capture_id": capture_id,
        "stage": "structured_complete",
        "status": resolved["status"],
        "field_count": len(fields),
        "missing_field_count": len(resolved["missing_fields"]),
        "conflict_field_count": len(resolved["conflict_fields"]),
        "structured_ms": round(elapsed_ms, 3),
        "updated_at": time.time(),
        "privacy": {"field_values_emitted": False},
    }
    _atomic_write(status_path, public_status, 0o644)
    return public_status


def _needs_recovery(input_path: Path, structured_path: Path) -> bool:
    try:
        input_capture = str(_load_json(input_path).get("capture_id") or "")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not input_capture:
        return False
    try:
        published_capture = str(_load_json(structured_path).get("capture_id") or "")
    except (OSError, ValueError, json.JSONDecodeError):
        published_capture = ""
    return input_capture != published_capture


def _notify_feedback(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    message = "%s %s %d %.3f" % (
        payload.get("capture_id") or "",
        payload.get("status") or "error",
        len(payload.get("fields") or {}),
        float((payload.get("timings") or {}).get("structured_ms") or 0.0),
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        try:
            sender.sendto(message.encode("ascii"), path)
        except OSError:
            # The receiver may already have stopped; private results remain available.
            pass


def process_pending(args: Any) -> None:
    try:
        if _needs_recovery(args.input, args.structured_output):
            status = process_once(
                args.input, args.schema, args.full_text_output,
                args.structured_output, args.status_output, args.schema_endpoint,
            )
            print(json.dumps({
                "event": "structured_complete",
                "status": status["status"],
                "field_count": status["field_count"],
                "structured_ms": status["structured_ms"],
            }), flush=True)
        if args.structured_output.is_file():
            _notify_feedback(str(args.socket) + ".result", _load_json(args.structured_output))
    except Exception as error:
        try:
            capture_id = str(_load_json(args.input).get("capture_id") or "")
        except (OSError, ValueError):
            capture_id = ""
        result = {
            "capture_id": capture_id, "status": "error", "fields": {},
            "error_type": type(error).__name__, "created_at": time.time(),
        }
        _atomic_write(args.structured_output, result, 0o600)
        _atomic_write(args.status_output, {
            "ok": False, "capture_id": capture_id, "status": "error",
            "stage": "structured_error", "error_type": type(error).__name__,
            "updated_at": time.time(),
        }, 0o644)
        _notify_feedback(str(args.socket) + ".result", result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=Path("/run/rk3568-camera/structured.sock"))
    parser.add_argument("--input", type=Path, default=Path("/run/rk3568-camera/native-ocr.json"))
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("/var/lib/rk3568-camera/active-field-rules.json"),
    )
    parser.add_argument(
        "--schema-endpoint",
        default="https://127.0.0.1:8443/internal/v1/field-rules",
    )
    parser.add_argument(
        "--full-text-output",
        type=Path,
        default=Path("/run/rk3568-camera/full-text-result.json"),
    )
    parser.add_argument(
        "--structured-output",
        type=Path,
        default=Path("/run/rk3568-camera/structured-result.json"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("/run/rk3568-camera/structured-status.json"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        process_once(
            args.input,
            args.schema,
            args.full_text_output,
            args.structured_output,
            args.status_output,
            args.schema_endpoint,
        )
        return 0

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.socket.unlink()
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    listener.bind(str(args.socket))
    os.chmod(str(args.socket), 0o660)
    try:
        process_pending(args)
        while True:
            listener.recv(4096)
            process_pending(args)
    finally:
        listener.close()
        try:
            args.socket.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
