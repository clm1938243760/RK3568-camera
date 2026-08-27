from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


class CameraRuntimeManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "runtime" / "manifest.json").read_text(encoding="utf-8")
        )

    def test_rk3568_port_is_explicitly_unverified(self) -> None:
        self.assertEqual(self.manifest["platform"], "rk3568")
        self.assertEqual(self.manifest["status"], "unverified")
        self.assertEqual(
            self.manifest["deployment"]["status"],
            "port_prepared_not_board_verified",
        )

    def test_detector_artifact_is_locked_without_false_board_benchmark(self) -> None:
        detector = self.manifest["paper_detector"]
        model = ROOT / detector["path"]

        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), detector["sha256"])
        self.assertEqual(detector["backend"], "opencv_dnn_cpu")
        self.assertNotIn("benchmark", detector)

    def test_service_keeps_the_rk3588_comparison_parameters(self) -> None:
        unit = (REPOSITORY / "systemd" / "rk3568-camera-report-trigger.service").read_text(
            encoding="utf-8"
        )

        self.assertIn("--text-only", unit)
        self.assertIn("--stable-seconds ${STABLE_SECONDS}", unit)
        self.assertIn("--burst-frames ${BURST_FRAMES}", unit)
        self.assertIn("--ocr-document-long-side ${OCR_DOCUMENT_LONG_SIDE}", unit)
        self.assertIn("--detector-backend opencv", unit)
        self.assertEqual(self.manifest["deployment"]["stable_seconds"], 0.5)
        self.assertEqual(self.manifest["deployment"]["burst_frames"], 2)
        self.assertEqual(self.manifest["deployment"]["ocr_document_long_side"], 3200)

    def test_rk3568_ppocr_is_serial_and_uses_target_specific_models(self) -> None:
        ppocr = self.manifest["ppocr"]
        execution = ppocr["recognition_execution"]

        self.assertEqual(ppocr["target_platform"], "rk3568")
        self.assertEqual(execution["parallel_contexts"], 1)
        self.assertEqual(execution["strategy"], "serialized_existing_service")
        self.assertRegex(ppocr["det_model_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(ppocr["rec_model_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
