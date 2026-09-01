from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "native_structured_worker.py"
SPEC = importlib.util.spec_from_file_location("native_structured_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class NativeStructuredWorkerTests(unittest.TestCase):
    def test_invalid_schema_does_not_fall_back_to_optional_defaults(self) -> None:
        with self.assertRaises(ValueError):
            WORKER._normalize_schema({})
        with self.assertRaises(ValueError):
            WORKER._normalize_schema({"fields": []})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                WORKER._load_schema(Path(directory) / "missing.json")

    def test_web_position_is_not_broadened_to_nearest_text(self) -> None:
        schema = WORKER._normalize_schema({"fields": [
            {"field_key": "patient_id", "label": "ID", "position": "below"}
        ]})
        self.assertEqual(schema[0]["relations"], ["same_text", "next_line_same_column"])

    def test_schema_error_is_terminal_and_duplicate_event_does_not_reprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                input=root / "input.json", schema=root / "missing.json",
                full_text_output=root / "full.json", structured_output=root / "result.json",
                status_output=root / "status.json", socket=root / "event.sock",
                schema_endpoint="",
            )
            args.input.write_text(json.dumps({
                "capture_id": "a" * 32, "image_size": [100, 100], "ocr": []
            }), encoding="utf-8")
            with mock.patch.object(WORKER, "_notify_feedback") as notify:
                WORKER.process_pending(args)
                result = json.loads(args.structured_output.read_text(encoding="utf-8"))
                self.assertEqual(result["capture_id"], "a" * 32)
                self.assertEqual(result["status"], "error")
                self.assertEqual(notify.call_args[0][1]["status"], "error")
                with mock.patch.object(WORKER, "process_once") as process:
                    WORKER.process_pending(args)
                    process.assert_not_called()

    def test_recovery_only_runs_for_unpublished_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "ocr.json"
            output_path = root / "structured.json"
            input_path.write_text('{"capture_id":"capture-a"}', encoding="utf-8")

            self.assertTrue(WORKER._needs_recovery(input_path, output_path))
            output_path.write_text('{"capture_id":"capture-a"}', encoding="utf-8")
            self.assertFalse(WORKER._needs_recovery(input_path, output_path))
            output_path.write_text('{"capture_id":"capture-b"}', encoding="utf-8")
            self.assertTrue(WORKER._needs_recovery(input_path, output_path))

    def test_old_web_schema_and_coordinates_generate_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "ocr.json"
            schema_path = root / "rules.json"
            full_text_path = root / "full.json"
            structured_path = root / "structured.json"
            status_path = root / "status.json"
            input_path.write_text(
                json.dumps(
                    {
                        "capture_id": "synthetic",
                        "image_size": [1000, 1000],
                        "source": {
                            "frame_size": {"width": 3840, "height": 2160},
                            "paper_corners": [[1, 1], [3838, 1], [3838, 2158], [1, 2158]],
                            "ocr_rotation": 90,
                            "ocr_document_long_side": 3200,
                        },
                        "timings": {
                            "stability_ms": 200.0,
                            "quality_ms": 70.0,
                            "transform_ms": 90.0,
                            "ocr_ms": 1234.5,
                            "paper_to_ocr_ms": 1594.5,
                        },
                        "ocr": [
                            {"text": "姓名", "score": 0.99, "box": [100, 100, 200, 140]},
                            {"text": "测试甲", "score": 0.98, "box": [220, 100, 320, 140]},
                            {"text": "卡号", "score": 0.97, "box": [100, 180, 200, 220]},
                            {"text": "P123456", "score": 0.96, "box": [220, 180, 360, 220]},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            schema_path.write_text(
                json.dumps(
                    {
                        "fields": [
                            {
                                "field_key": "patient_name",
                                "label": "姓名",
                                "char_type": "any",
                                "min_ocr_score": 0.9,
                                "max_distance": 180,
                                "required": True,
                            },
                            {
                                "field_key": "patient_id",
                                "label": "卡号",
                                "char_type": "alnum",
                                "min_ocr_score": 0.9,
                                "max_distance": 180,
                                "required": True,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            public = WORKER.process_once(
                input_path,
                schema_path,
                full_text_path,
                structured_path,
                status_path,
            )
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
            full_text = json.loads(full_text_path.read_text(encoding="utf-8"))
            document = full_text["document"]

            self.assertEqual(public["status"], "accepted")
            self.assertEqual(public["field_count"], 4)
            self.assertEqual(structured["status"], "accepted")
            self.assertEqual(full_text["source"]["ocr_rotation"], 90)
            self.assertEqual(document["line_count"], 2)
            self.assertEqual(document["blocks"][0]["normalized_box"], [100, 100, 200, 140])
            self.assertEqual(
                structured["fields"]["patient_id"]["source_span_ids"], [3, 4]
            )
            self.assertAlmostEqual(
                structured["fields"]["patient_id"]["probability"], 0.96
            )
            self.assertGreaterEqual(structured["timings"]["total_ms"], 1594.5)
            self.assertEqual(structured["timings"]["stability_ms"], 200.0)
            self.assertEqual(
                set(structured["fields"]["patient_id"]),
                {
                    "value",
                    "probability",
                    "source_span_ids",
                    "matched_prompt",
                    "relation",
                },
            )

    def test_low_confidence_required_field_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ocr.json").write_text(
                json.dumps(
                    {
                        "capture_id": "low-score",
                        "image_size": [1000, 1000],
                        "ocr": [
                            {"text": "姓名", "score": 0.99, "box": [100, 100, 200, 140]},
                            {"text": "测试乙", "score": 0.50, "box": [220, 100, 320, 140]},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "rules.json").write_text(
                json.dumps(
                    {
                        "fields": [
                            {
                                "field_key": "patient_name",
                                "label": "姓名",
                                "min_ocr_score": 0.9,
                                "required": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            public = WORKER.process_once(
                root / "ocr.json",
                root / "rules.json",
                root / "full.json",
                root / "structured.json",
                root / "status.json",
            )
            self.assertEqual(public["status"], "rejected")
            self.assertEqual(public["missing_field_count"], 1)
            self.assertEqual(public["field_count"], 0)


if __name__ == "__main__":
    unittest.main()
