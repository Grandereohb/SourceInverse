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


if __name__ == "__main__":
    unittest.main()
