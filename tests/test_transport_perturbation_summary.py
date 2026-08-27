import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_transport_perturbation_matrix import summarize


class TransportPerturbationSummaryTests(unittest.TestCase):
    def _write_matrix(self, root: Path, label: str, wind_scale: float, x_m: float):
        scenario_id = "scenario_1"
        plan = {
            "batch_manifest": {"sha256": "BATCH"},
            "scenarios": [{"scenario_id": scenario_id}],
            "solvers": ["production"],
            "epochs": 10,
            "seed": 0,
            "q_mode": "smooth_time",
            "source_position_mode": "single",
            "physics_settings": {
                "wind_scale": wind_scale,
                "decay_per_hour": 0.5,
                "d_min_m2s": 1.0,
                "sigma_src_norm": 0.05,
                "wind_vector_smoothing": "enabled",
            },
        }
        root.mkdir()
        (root / "matrix_experiment_plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        run_root = root / scenario_id / "run"
        run_root.mkdir(parents=True)
        quality = {
            "source": {"initialization": {"x_m": 1.0, "y_m": 2.0}},
            "fit_raw_rmse": 3.0,
        }
        quality_path = run_root / "result_quality_report.json"
        quality_path.write_text(json.dumps(quality), encoding="utf-8")
        manifest = {
            "runtime": {
                "recurrent_solver": "production",
                "run_id": "heuristic",
                "training_wall_time_s": 4.0,
            },
            "source": {"x_m": x_m, "y_m": 0.0},
            "checkpoint": {"best_raw_loss": 2.0},
            "inputs": {
                "sites": {"sha256": "S"},
                "concentration": {"sha256": "C"},
                "wind": {"sha256": "W"},
            },
        }
        (run_root / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_summary_computes_pairwise_and_baseline_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            specs = []
            for label, scale, x_m in (
                ("w020", 0.20, -2.0),
                ("w025", 0.25, 0.0),
                ("w030", 0.30, 3.0),
            ):
                matrix_root = root / label
                self._write_matrix(matrix_root, label, scale, x_m)
                specs.append((label, matrix_root))
            truth_path = root / "truth.json"
            truth_path.write_text(
                json.dumps(
                    {
                        "success_threshold_m": 500.0,
                        "scenarios": [
                            {
                                "scenario_id": "scenario_1",
                                "selected_localization_error_m": 600.0,
                                "selected_success": False,
                                "failure_category": "reachability_failure",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = summarize(specs, truth_path, baseline_label="w025")

            row = result["scenarios"][0]
            self.assertEqual(row["wind_scale_max_pairwise_source_drift_m"], 5.0)
            self.assertEqual(row["wind_scale_max_baseline_source_drift_m"], 3.0)
            self.assertEqual(row["selected_localization_error_m"], 600.0)
            self.assertEqual(row["total_perturbation_training_wall_time_s"], 12.0)


if __name__ == "__main__":
    unittest.main()
