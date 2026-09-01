import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = load("benchmark_ocr_http")
COMPARE = load("compare_benchmarks")
COLLECT = load("collect_e2e_timings")


class BenchmarkToolTests(unittest.TestCase):
    def test_percentile_and_summary_are_deterministic(self):
        summary = BENCHMARK.summarize([10, 20, 30, 40])

        self.assertEqual(summary["median"], 25.0)
        self.assertEqual(summary["p95"], 38.5)
        self.assertEqual(summary["count"], 4)

    def test_comparison_rejects_no_metric_by_returning_empty_rows(self):
        self.assertEqual(COMPARE.metric_rows({}, {}), [])

    def test_comparison_reports_rk3568_over_rk3588_ratio(self):
        left = {"metrics": {"client_elapsed_ms": {"median": 100.0}}}
        right = {"metrics": {"client_elapsed_ms": {"median": 250.0}}}

        row = COMPARE.metric_rows(left, right)[0]

        self.assertEqual(row["delta_ms"], 150.0)
        self.assertEqual(row["rk3568_over_rk3588"], 2.5)

    def test_live_collector_records_timing_without_ocr_text(self):
        collector = COLLECT.TimingCollector("rk3568")
        collector.observe_status(
            {"updated_at": 100.0, "paper_detected": True, "capture_id": ""}
        )
        collector.observe_status(
            {"updated_at": 100.5, "paper_detected": True, "capture_id": "capture-a"}
        )
        with mock.patch.object(COLLECT.time, "time", return_value=102.0):
            record = collector.observe_result(
                {
                    "capture_id": "capture-a",
                    "status": "accepted",
                    "timings": {"ocr_ms": 800.0, "total_ms": 1200.0},
                    "source": {"ocr_item_count": 10},
                    "fields": {"patient_id": {"value": "must not be retained"}},
                }
            )

        self.assertEqual(record["paper_to_result_ms"], 2000.0)
        self.assertEqual(record["ocr_ms"], 800.0)
        self.assertNotIn("must not be retained", str(collector.output()))
        self.assertNotIn("p95", collector.output()["summary"]["total_ms"])


if __name__ == "__main__":
    unittest.main()
