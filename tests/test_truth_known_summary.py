import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_truth_known_multistart.py"
SPEC = importlib.util.spec_from_file_location("truth_known_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TruthKnownSummaryTests(unittest.TestCase):
    def test_average_ranks_handle_ties(self):
        self.assertEqual(MODULE._average_ranks([2.0, 1.0, 2.0]), [2.5, 1.0, 2.5])

    def test_solver_summary_reports_selection_regret(self):
        rows = [
            {
                "start_id": "low_loss_wrong",
                "best_raw_loss": 10.0,
                "localization_error_m": 400.0,
                "final_x_m": 0.0,
                "final_y_m": 0.0,
                "training_wall_time_s": 1.0,
            },
            {
                "start_id": "higher_loss_right",
                "best_raw_loss": 10.2,
                "localization_error_m": 100.0,
                "final_x_m": 100.0,
                "final_y_m": 0.0,
                "training_wall_time_s": 2.0,
            },
        ]
        summary = MODULE._solver_summary(rows)
        self.assertEqual(summary["selected_start_id"], "low_loss_wrong")
        self.assertEqual(summary["oracle_start_id"], "higher_loss_right")
        self.assertEqual(summary["selection_regret_m"], 300.0)
        self.assertEqual(summary["near_optimal_count"], 2)
        self.assertEqual(summary["near_optimal_max_source_spread_m"], 100.0)

    def test_baseline_truth_join_is_posthoc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [
                            {"method": "B0", "source_x_m": 3.0, "source_y_m": 4.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evaluated = MODULE._evaluate_baselines(path, (0.0, 0.0))
        self.assertEqual(evaluated["results"][0]["localization_error_m"], 5.0)
        self.assertIn("post-optimization", evaluated["truth_join_timing"])


if __name__ == "__main__":
    unittest.main()
