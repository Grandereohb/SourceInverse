import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_multistart_matrix_truth import aggregate


class AggregateMultistartMatrixTruthTests(unittest.TestCase):
    def _write_summary(
        self,
        root: Path,
        scenario_id: str,
        selected_error: float,
        oracle_error: float,
        closest_initial_error: float = 25.0,
    ) -> None:
        scenario_root = root / scenario_id
        scenario_root.mkdir(parents=True)
        selected_run_root = scenario_root / "selected_run"
        selected_run_root.mkdir()
        quality_path = selected_run_root / "result_quality_report.json"
        quality_path.write_text(
            json.dumps(
                {
                    "source": {"min_boundary_margin_m": 750.0},
                    "recurrent_pde": {
                        "substep_cap_hit_count": 0,
                        "source_quadrature_cap_hit_count": 0,
                    },
                    "quality_diagnostics": {
                        "dominant_station": {"residual_energy_ratio": 0.6}
                    },
                    "is_reasonable": False,
                }
            ),
            encoding="utf-8",
        )
        payload = {
            "truth_join_timing": "truth is read only by this post-optimization summarizer",
            "scenario": {"id": scenario_id},
            "solver_summaries": {
                "production": {
                    "selected_start_id": "selected",
                    "selected_localization_error_m": selected_error,
                    "selected_best_raw_loss": 2.0,
                    "oracle_start_id": "oracle",
                    "oracle_localization_error_m": oracle_error,
                    "selection_regret_m": selected_error - oracle_error,
                    "second_best_loss_relative_gap": 0.2,
                    "near_optimal_count": 1,
                    "near_optimal_max_source_spread_m": 0.0,
                    "spearman_loss_vs_localization_error": 0.5,
                    "total_training_wall_time_s": 12.0,
                }
            },
            "runs": [
                {
                    "solver": "production",
                    "start_id": "selected",
                    "initial_error_m": 50.0,
                    "fit_raw_rmse": 1.5,
                    "optimizer_displacement_m": 100.0,
                    "warning_count": 2,
                    "run_dir": str(selected_run_root),
                    "quality_report_sha256": hashlib.sha256(
                        quality_path.read_bytes()
                    ).hexdigest().upper(),
                },
                {
                    "solver": "production",
                    "start_id": "oracle",
                    "initial_error_m": closest_initial_error,
                    "fit_raw_rmse": 2.0,
                    "optimizer_displacement_m": 80.0,
                    "warning_count": 1,
                }
            ],
        }
        (scenario_root / "truth_known_multistart_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_aggregate_separates_reachability_and_selection_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "matrix_plan.json"
            scenario_ids = ["success", "selection", "reachability"]
            plan_path.write_text(
                json.dumps(
                    {
                        "scenario_count": 3,
                        "scenarios": [
                            {"scenario_id": scenario_id}
                            for scenario_id in scenario_ids
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self._write_summary(root, "success", 400.0, 300.0, 0.0)
            self._write_summary(root, "selection", 700.0, 200.0)
            self._write_summary(root, "reachability", 900.0, 800.0)

            result = aggregate(plan_path, root, success_threshold_m=500.0)

            self.assertEqual(result["aggregate"]["selected_success_count"], 1)
            self.assertEqual(result["aggregate"]["oracle_success_count"], 2)
            self.assertEqual(result["aggregate"]["truth_coincident_start_count"], 1)
            self.assertEqual(
                result["sensitivity_excluding_truth_coincident_starts"][
                    "scenario_count"
                ],
                2,
            )
            self.assertEqual(
                result["aggregate"]["failure_category_counts"],
                {
                    "selected_success": 1,
                    "selection_failure_reachable": 1,
                    "reachability_failure": 1,
                },
            )
            self.assertEqual(
                result["aggregate"]["selected_localization_error_m"]["median"],
                700.0,
            )
            self.assertEqual(
                [row["failure_category"] for row in result["scenarios"]],
                [
                    "selected_success",
                    "selection_failure_reachable",
                    "reachability_failure",
                ],
            )

    def test_missing_expected_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "matrix_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "scenario_count": 1,
                        "scenarios": [{"scenario_id": "missing"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                aggregate(plan_path, root)


if __name__ == "__main__":
    unittest.main()
