import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_comparison_profile_matches_rk3588_baseline(self):
        values = {}
        for line in (ROOT / "config" / "camera.env.example").read_text(
            encoding="utf-8"
        ).splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value

        self.assertEqual(values["STABLE_SECONDS"], "0.5")
        self.assertEqual(values["BURST_FRAMES"], "2")
        self.assertEqual(values["OCR_DOCUMENT_LONG_SIDE"], "3200")
        self.assertEqual(values["OCR_ENDPOINT"], "http://127.0.0.1:5002/ocr")

    def test_installer_does_not_manage_existing_board_services(self):
        script = (ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertNotIn("systemctl restart rk3568-ppocr", script)
        self.assertNotIn("systemctl restart rk3568-patient-frame", script)
        self.assertNotIn("usb_gadget", script)
        self.assertIn("--activate", script)

        snapshot_unit = (
            ROOT / "systemd" / "rk3568-camera-snapshot-bridge.service"
        ).read_text(encoding="utf-8")
        trigger_unit = (
            ROOT / "systemd" / "rk3568-camera-report-trigger.service"
        ).read_text(encoding="utf-8")
        monitor_unit = (
            ROOT / "systemd" / "rk3568-camera-ocr-overlay.service"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Wants=network-online.target rk3568-patient-frame", snapshot_unit)
        self.assertNotIn("Wants=network-online.target rk3568-ppocr", trigger_unit)
        self.assertNotIn("Wants=network-online.target rk3568-camera-report-trigger", monitor_unit)
        self.assertIn("/tmp/rk3568_camera_ocr_%%d.jpg", snapshot_unit)
        self.assertNotIn("/tmp/rk3568_camera_ocr_%d.jpg", snapshot_unit)

    def test_native_installer_has_safe_activation_and_single_job_default(self):
        script = (ROOT / "scripts" / "install_native_pipeline.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('NATIVE_BUILD_JOBS="${NATIVE_BUILD_JOBS:-1}"', script)
        self.assertIn("restore_previous_pipeline", script)
        self.assertIn("Native activation failed", script)
        self.assertNotIn("rm -rf /etc/systemd/system", script)

    def test_manifest_is_rk3568_and_not_claimed_verified(self):
        manifest = json.loads(
            (ROOT / "report_parser" / "runtime" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(manifest["platform"], "rk3568")
        self.assertEqual(manifest["status"], "unverified")
        self.assertEqual(manifest["ppocr"]["recognition_execution"]["parallel_contexts"], 1)
        self.assertEqual(manifest["deployment"]["detector_backend"], "onnxruntime")
        self.assertEqual(manifest["paper_detector"]["backend"], "onnxruntime_cpu")

    def test_monitor_uses_rk3568_jpeg_preview_instead_of_rk3588_webrtc(self):
        monitor = (ROOT / "camera_ocr_overlay.py").read_text(encoding="utf-8")

        self.assertIn('const endpoint="/api/frame.jpg"', monitor)
        self.assertIn("frame endpoint must be loopback HTTP", monitor)
        self.assertNotIn(":8891/camera/", monitor)
        self.assertNotIn("MediaMTXWebRTCReader", monitor)

    def test_python37_production_syntax_gate(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_python37_syntax.py")],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_parser_protocol_types_use_python37_compatibility_layer(self):
        package = ROOT / "report_parser" / "src" / "rk3588_report_parser"
        for path in package.glob("*.py"):
            if path.name == "typing_compat.py":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("from typing import"):
                    self.assertNotIn("Protocol", line, str(path))

        compat = (package / "typing_compat.py").read_text(encoding="utf-8")
        self.assertIn("from typing import Protocol", compat)
        self.assertIn("class Protocol(object)", compat)

    def test_rk3568_python_runtime_versions_are_pinned(self):
        requirements = (
            ROOT / "report_parser" / "requirements-camera-trigger-board.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("numpy==1.21.6", requirements)
        self.assertIn("onnxruntime==1.14.0", requirements)
        self.assertNotIn("1.17.3", requirements)


if __name__ == "__main__":
    unittest.main()
