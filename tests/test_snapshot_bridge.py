import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rk3568_snapshot_bridge",
    ROOT / "scripts" / "rk3568_snapshot_bridge.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def jpeg(trailer=b""):
    return b"\xff\xd8" + (b"x" * 4096) + b"\xff\xd9" + trailer


class SnapshotBridgeTests(unittest.TestCase):
    def test_normalize_jpeg_accepts_small_trailer_and_trims_it(self):
        normalized = MODULE.normalize_jpeg(jpeg(b"trailer"))

        self.assertTrue(normalized.startswith(b"\xff\xd8"))
        self.assertTrue(normalized.endswith(b"\xff\xd9"))
        self.assertNotIn(b"trailer", normalized)

    def test_normalize_jpeg_rejects_incomplete_frame(self):
        with self.assertRaises(ValueError):
            MODULE.normalize_jpeg(b"not-a-jpeg", minimum_bytes=1)

    def test_publish_rotates_slots_and_writes_non_sensitive_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge = MODULE.SnapshotBridge(
                "http://127.0.0.1/frame.jpg",
                str(root / "frame_%d.jpg"),
                root / "status.json",
                slots=2,
                minimum_bytes=1,
            )

            first = bridge.publish(jpeg(), 12.5, now=100.0)
            second = bridge.publish(jpeg(), 10.0, now=101.0)
            third = bridge.publish(jpeg(), 9.0, now=102.0)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))

            self.assertEqual([first["slot"], second["slot"], third["slot"]], [0, 1, 0])
            self.assertEqual(status["published_frames"], 3)
            self.assertNotIn("image", status)
            self.assertTrue((root / "frame_0.jpg").read_bytes().endswith(b"\xff\xd9"))


if __name__ == "__main__":
    unittest.main()
