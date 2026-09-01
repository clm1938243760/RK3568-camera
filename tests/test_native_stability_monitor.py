import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "monitor_native_stability.py"
SPEC = importlib.util.spec_from_file_location("monitor_native_stability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


class NativeStabilityMonitorTests(unittest.TestCase):
    def test_stable_single_process_is_accepted(self):
        result = MONITOR.summarize(
            [
                {
                    "pid": 10,
                    "rss_kib": 100,
                    "fd_count": 8,
                    "threads": 4,
                    "stage": "waiting_paper",
                    "status_age_ms": 20.0,
                },
                {
                    "pid": 10,
                    "rss_kib": 105,
                    "fd_count": 8,
                    "threads": 4,
                    "stage": "waiting_paper",
                    "status_age_ms": 30.0,
                },
            ],
            [],
            1.0,
            11.0,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["fd_count"]["growth"], 0)

    def test_restart_or_stale_status_is_rejected(self):
        samples = [
            {
                "pid": 10,
                "rss_kib": 100,
                "fd_count": 8,
                "threads": 4,
                "stage": "waiting_paper",
                "status_age_ms": 20.0,
            },
            {
                "pid": 11,
                "rss_kib": 100,
                "fd_count": 8,
                "threads": 4,
                "stage": "waiting_paper",
                "status_age_ms": 2500.0,
            },
        ]

        self.assertFalse(MONITOR.summarize(samples, [], 1.0, 11.0)["ok"])


if __name__ == "__main__":
    unittest.main()
