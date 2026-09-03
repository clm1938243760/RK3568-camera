import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "camera_ocr_overlay.py"
SPEC = importlib.util.spec_from_file_location("camera_ocr_overlay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CameraOverlayConfigurationTest(unittest.TestCase):
    def make_store(self, root: Path, restarts):
        rules = root / "rules.json"
        rules.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "fields": [
                        {
                            "type": "selected_identifier",
                            "lengths": [16],
                            "charset": "alphanumeric",
                            "enabled": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return MODULE.CaptureConfigurationStore(
            root / "settings.json",
            rules,
            root / "camera.env",
            restart_trigger=lambda: restarts.append(True),
        )

    def test_region_is_persisted_and_exported_to_native_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restarts = []
            store = self.make_store(root, restarts)
            (root / "camera.env").write_text(
                "FRAME_ENDPOINT=http://127.0.0.1:8090/api/frame.jpg?quality=95\n"
                "OCR_ROTATION=90\n",
                encoding="utf-8",
            )
            result = store.update(
                {
                    "display_rotation": 0,
                    "ocr_rotation": 0,
                    "match": {"length": 16, "charset": "alphanumeric"},
                    "patient_query_enabled": False,
                    "auto_entry_enabled": False,
                    "recognition_region": {
                        "top": 0.2,
                        "bottom": 0.7,
                        "canonical_long_side": 3072,
                    },
                }
            )
            self.assertEqual(
                result["recognition_region"],
                {"top": 0.2, "bottom": 0.7, "canonical_long_side": 3072},
            )
            environment = (root / "camera.env").read_text(encoding="utf-8")
            self.assertIn("OCR_ROTATION=90\n", environment)
            self.assertIn("OCR_REGION_TOP=0.200000\n", environment)
            self.assertIn("OCR_REGION_BOTTOM=0.700000\n", environment)
            self.assertIn("OCR_DOCUMENT_LONG_SIDE=3072\n", environment)
            self.assertIn(
                "FRAME_ENDPOINT=http://127.0.0.1:8090/api/frame.jpg?quality=95\n",
                environment,
            )
            self.assertEqual(len(restarts), 1)

    def test_invalid_region_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root, [])
            with self.assertRaises(ValueError):
                store.update(
                    {
                        "display_rotation": 0,
                        "ocr_rotation": 0,
                        "match": {"length": 16, "charset": "alphanumeric"},
                        "patient_query_enabled": False,
                        "auto_entry_enabled": False,
                        "recognition_region": {
                            "top": 0.8,
                            "bottom": 0.3,
                            "canonical_long_side": 3200,
                        },
                    }
                )
            self.assertFalse((root / "settings.json").exists())

    def test_existing_native_environment_is_the_initial_source_of_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root, [])
            (root / "camera.env").write_text(
                "OCR_ROTATION=90\n"
                "OCR_REGION_TOP=0.000000\n"
                "OCR_REGION_BOTTOM=1.000000\n"
                "OCR_DOCUMENT_LONG_SIDE=3200\n",
                encoding="utf-8",
            )
            self.assertEqual(
                store.snapshot()["recognition_region"],
                {"top": 0.0, "bottom": 1.0, "canonical_long_side": 3200},
            )

    def test_full_text_overlay_preserves_native_recognition_region(self):
        payload = {
            "status": "accepted",
            "capture_id": "capture-1",
            "source": {
                "frame_size": {"width": 3840, "height": 2160},
                "paper_corners": [[100, 100], [900, 100], [900, 1900], [100, 1900]],
                "ocr_rotation": 90,
                "ocr_document_long_side": 3200,
                "recognition_region": {
                    "crop_normalized": [0.0, 0.13, 1.0, 0.60],
                },
            },
            "document": {
                "schema_version": 2,
                "image_size": [1176, 1504],
                "blocks": [
                    {
                        "id": 1,
                        "source_index": 0,
                        "line_id": 1,
                        "text": "field",
                        "box": [0, 0, 100, 100],
                        "normalized_box": [0, 0, 85, 66],
                        "normalized_polygon": [[0, 0], [85, 0], [85, 66], [0, 66]],
                        "score": 0.99,
                    }
                ],
            },
        }

        document = MODULE._full_text_document(payload, "capture-1")

        self.assertTrue(document["available"])
        self.assertEqual(
            document["source"]["recognition_region"]["crop_normalized"],
            [0.0, 0.13, 1.0, 0.60],
        )

    def test_overlay_maps_cropped_y_back_to_canonical_document(self):
        self.assertIn(
            b"const documentY=top+y*(bottom-top);",
            MODULE.PAGE,
        )

    def test_display_state_maps_native_pipeline_stages(self):
        cases = (
            ("waiting_paper", "absent", "wait_scan"),
            ("detecting_stability", "tracking", "report_detecting"),
            ("checking_quality", "locked", "report_detecting"),
            ("ocr_running", "ocr_primary", "report_detecting"),
            ("structuring", "structuring", "report_detecting"),
            ("structured_complete", "completed", "entry_completed"),
            ("structured_rejected", "reposition_required", "paper_reposition"),
        )
        for stage, capture_stage, expected_screen in cases:
            with self.subTest(stage=stage):
                payload = MODULE.display_state_payload(
                    {
                        "stage": stage,
                        "capture_stage": capture_stage,
                        "service_state": "active",
                    }
                )
                self.assertEqual(payload["display"]["screen"], expected_screen)

    def test_normalized_status_preserves_native_stage(self):
        payload = MODULE.normalize_trigger_status({"stage": "ocr_running"})

        self.assertEqual(payload["stage"], "ocr_running")


if __name__ == "__main__":
    unittest.main()
