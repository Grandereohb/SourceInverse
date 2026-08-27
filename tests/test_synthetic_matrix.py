import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_synthetic_matrix.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SyntheticMatrixTests(unittest.TestCase):
    def test_matrix_rejects_duplicate_scenario_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenarios": [
                            {"scenario_id": "same"},
                            {"scenario_id": "same"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                MODULE.load_matrix(path)

    def test_factorial_expansion_is_deterministic_and_carries_factors(self):
        factorial = {
            "scenario_prefix": "case",
            "base_seed": 100,
            "input_ids": ["layout_a", "layout_b"],
            "positions": [
                {"id": "center", "fraction_x": 0.5, "fraction_y": 0.5}
            ],
            "physics_conditions": [
                {
                    "id": "matched",
                    "diffusivity": 2.0,
                    "sigma": 100.0,
                    "decay": 0.1,
                    "wind_factor": 0.25,
                }
            ],
            "defaults": {"q_shape": "constant"},
        }
        rows = MODULE._expand_factorial(factorial)
        self.assertEqual([row["scenario_id"] for row in rows], ["case_0001", "case_0002"])
        self.assertEqual([row["seed"] for row in rows], [101, 102])
        self.assertEqual(rows[1]["design_factors"]["input_id"], "layout_b")
        self.assertEqual(rows[0]["design_factors"]["physics_condition_id"], "matched")

    def test_factorial_crosses_release_conditions_without_changing_legacy_designs(self):
        factorial = {
            "scenario_prefix": "test",
            "base_seed": 500,
            "input_ids": ["layout"],
            "positions": [
                {"id": "center", "fraction_x": 0.5, "fraction_y": 0.5}
            ],
            "physics_conditions": [
                {
                    "id": "matched",
                    "diffusivity": 2.0,
                    "sigma": 100.0,
                    "decay": 0.1,
                    "wind_factor": 0.25,
                }
            ],
            "release_conditions": [
                {"id": "constant", "q_shape": "constant"},
                {"id": "double_peak", "q_shape": "double_peak"},
            ],
            "defaults": {"target_peak": 100.0},
        }

        rows = MODULE._expand_factorial(factorial)

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["seed"] for row in rows], [501, 502])
        self.assertEqual(
            [row["design_factors"]["release_condition_id"] for row in rows],
            ["constant", "double_peak"],
        )
        self.assertEqual([row["q_shape"] for row in rows], ["constant", "double_peak"])

    def test_frozen_sumitomo_ood_design_expands_to_90_unique_scenarios(self):
        design_path = (
            ROOT
            / "experiments"
            / "reliability_audit_test_ood_sumitomo_v1.json"
        )

        payload = MODULE.load_matrix(design_path)
        rows = payload["scenarios"]

        self.assertEqual(len(rows), 90)
        self.assertEqual(len({row["scenario_id"] for row in rows}), 90)
        self.assertEqual(len({row["seed"] for row in rows}), 90)
        self.assertEqual(
            {row["design_factors"]["release_condition_id"] for row in rows},
            {"constant_release", "double_peak_release"},
        )
        self.assertEqual(payload["status"], "frozen_unrun")


if __name__ == "__main__":
    unittest.main()
