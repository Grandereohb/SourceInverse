import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_selective_diagnostics import evaluate, load_feature_config


class SelectiveDiagnosticTests(unittest.TestCase):
    def test_tied_scores_are_not_split_and_direction_is_respected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "aggregate.json"
            input_path.write_text(
                json.dumps(
                    {
                        "success_threshold_m": 500.0,
                        "scenarios": [
                            {
                                "scenario_id": "a",
                                "selected_localization_error_m": 100.0,
                                "gap": 0.9,
                            },
                            {
                                "scenario_id": "b",
                                "selected_localization_error_m": 900.0,
                                "gap": 0.9,
                            },
                            {
                                "scenario_id": "c",
                                "selected_localization_error_m": 800.0,
                                "gap": 0.2,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate(input_path, features={"gap": "descending"})

            curve = result["curves"]["gap"]
            self.assertEqual([row["accepted_count"] for row in curve], [2, 3])
            self.assertEqual(curve[0]["coverage"], 2 / 3)
            self.assertEqual(curve[0]["failure_risk"], 0.5)
            self.assertEqual(curve[0]["accepted_scenario_ids"], ["a", "b"])

    def test_unknown_direction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "aggregate.json"
            input_path.write_text(
                json.dumps(
                    {
                        "success_threshold_m": 500.0,
                        "scenarios": [
                            {
                                "scenario_id": "a",
                                "selected_localization_error_m": 100.0,
                                "score": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                evaluate(input_path, features={"score": "sideways"})

    def test_feature_config_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "features.json"
            config_path.write_text(
                json.dumps(
                    {
                        "failure_threshold_m": 500.0,
                        "features": [
                            {"name": "gap", "direction": "descending"},
                            {"name": "gap", "direction": "ascending"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_feature_config(config_path)


if __name__ == "__main__":
    unittest.main()
