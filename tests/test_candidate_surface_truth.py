import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_gaussian_candidate_surfaces_truth import evaluate


class CandidateSurfaceTruthTests(unittest.TestCase):
    def test_relocated_surface_is_joined_to_truth_only_during_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "relocated_dataset"
            scenario_root = dataset_root / "scenario_001"
            scenario_root.mkdir(parents=True)
            scenario_path = scenario_root / "scenario_manifest.json"
            scenario_path.write_text(
                json.dumps({"source": {"x_m": 0.0, "y_m": 0.0}}),
                encoding="utf-8",
            )

            analysis_root = dataset_root / "analysis" / "surfaces"
            surface_root = analysis_root / "scenario_001"
            surface_root.mkdir(parents=True)
            surface_path = surface_root / "candidate_surfaces.json"
            surface_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "method": "test_method",
                                "source_x_m": 1000.0,
                                "source_y_m": 0.0,
                                "anomaly_rmse": 1.0,
                                "forward_evaluation_count": 2,
                                "wall_time_s": 0.1,
                                "candidates": [
                                    {
                                        "source_x_m": 1000.0,
                                        "source_y_m": 0.0,
                                        "anomaly_rmse": 1.0,
                                    },
                                    {
                                        "source_x_m": 100.0,
                                        "source_y_m": 0.0,
                                        "anomaly_rmse": 2.0,
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            status_path = analysis_root / "execution_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "outputs": [
                            {
                                "scenario_id": "scenario_001",
                                "scenario_manifest_path": "D:/old/scenario_manifest.json",
                                "surface_path": "D:/old/candidate_surfaces.json",
                                "design_factors": {
                                    "release_condition_id": "double_peak_release"
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate(
                status_path,
                threshold_m=500.0,
                dataset_root=dataset_root,
                resource_accounting="shared_per_surface",
            )

            summary = result["method_summaries"]["test_method"]
            self.assertEqual(summary["selected_success_count"], 0)
            self.assertEqual(summary["oracle_success_count"], 1)
            self.assertEqual(
                summary["failure_categories"]["selection_failure_reachable"], 1
            )
            self.assertEqual(result["rows"][0]["selection_regret_m"], 900.0)
            self.assertEqual(
                result["rows"][0]["release_condition_id"],
                "double_peak_release",
            )
            self.assertEqual(
                result["shared_surface_totals"]["total_forward_evaluation_count"],
                2,
            )
            self.assertEqual(
                result["shared_surface_totals"]["scenario_count"], 1
            )


if __name__ == "__main__":
    unittest.main()
