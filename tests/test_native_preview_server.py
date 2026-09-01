import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "native_preview_server.py"
SPEC = importlib.util.spec_from_file_location("native_preview_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NativePreviewServerTest(unittest.TestCase):
    def test_pipeline_uses_secondary_nv12_path_and_mpp_encoder(self):
        pipeline = MODULE.build_pipeline_description("/dev/video1", 1920, 1080, 30, 5, 8)
        self.assertIn("video/x-raw,format=NV12", pipeline)
        self.assertIn("framerate=5/1", pipeline)
        self.assertIn("mppjpegenc", pipeline)
        self.assertNotIn("videoconvert", pipeline)

    def test_pipeline_rejects_unsafe_or_impossible_settings(self):
        with self.assertRaises(ValueError):
            MODULE.build_pipeline_description("/tmp/video1", 1920, 1080, 30, 5, 8)
        with self.assertRaises(ValueError):
            MODULE.build_pipeline_description("/dev/video1", 1920, 1080, 5, 15, 8)

    def test_latest_jpeg_rejects_partial_frames_and_tracks_age(self):
        latest = MODULE.LatestJpeg()
        self.assertFalse(latest.publish(b"not-a-jpeg", now=10.0))
        self.assertTrue(latest.publish(b"\xff\xd8data\xff\xd9", now=10.0))
        value = latest.snapshot(now=10.125)
        self.assertEqual(value["sequence"], 1)
        self.assertEqual(value["data"], b"\xff\xd8data\xff\xd9")
        self.assertAlmostEqual(value["age_ms"], 125.0)


if __name__ == "__main__":
    unittest.main()
