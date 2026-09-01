import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NativeRuntimeManifestTests(unittest.TestCase):
    def test_native_candidate_contract_is_locked(self):
        manifest = json.loads(
            (ROOT / "native" / "runtime-manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["platform"], "rk3568")
        self.assertEqual(manifest["capture"]["pixel_format"], "NV12")
        self.assertEqual(manifest["pipeline"]["final_frame_count"], 1)
        self.assertEqual(manifest["pipeline"]["retry_count"], 0)
        self.assertEqual(manifest["models"]["ppocr_detection"]["precision"], "fp16")
        self.assertEqual(manifest["models"]["ppocr_recognition"]["batch"], 1)
        self.assertTrue(manifest["validation"]["rollback_rehearsed"])


if __name__ == "__main__":
    unittest.main()
