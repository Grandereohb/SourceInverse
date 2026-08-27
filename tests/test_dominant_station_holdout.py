import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_dominant_station_holdout import prepare_holdout  # noqa: E402


class DominantStationHoldoutTests(unittest.TestCase):
    def test_selects_largest_positive_residual_energy_without_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concentration = root / "concentration.csv"
            pd.DataFrame(
                {
                    "time": ["t0", "t1"],
                    "a": [0.0, 0.0],
                    "b": [1.0, 1.0],
                    "dominant": [10.0, 12.0],
                    "TARGET_POLLUTANT": ["x", "x"],
                }
            ).to_csv(concentration, index=False)
            multistart = root / "multistart"
            multistart.mkdir()
            (multistart / "experiment_plan.json").write_text(
                json.dumps(
                    {
                        "source_domain_m": {
                            "x_min_m": -100,
                            "x_max_m": 100,
                            "y_min_m": -100,
                            "y_max_m": 100,
                        }
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            run_manifest = root / "run_manifest.json"
            run_manifest.write_text(
                json.dumps({"source": {"x_m": 20.0, "y_m": -40.0}}),
                encoding="utf-8",
            )
            design_path = root / "profile.json"
            design_path.write_text(
                json.dumps(
                    {
                        "provenance": {
                            "selected_start_id": "selected",
                            "selected_checkpoint_path": str(checkpoint),
                            "selected_checkpoint_sha256": "HASH",
                            "input_run_manifests": [
                                {
                                    "start_id": "selected",
                                    "manifest_path": str(run_manifest),
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            training, design, manifest = prepare_holdout(
                concentration, multistart, design_path
            )
            self.assertEqual(manifest["heldout_station"], "dominant")
            self.assertNotIn("dominant", training.columns)
            start = design["starts"][0]
            self.assertAlmostEqual(start["fraction_x"], 0.6)
            self.assertAlmostEqual(start["fraction_y"], 0.3)


if __name__ == "__main__":
    unittest.main()
