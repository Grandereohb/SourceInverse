import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_local_profile_design import generate_design  # noqa: E402


class LocalProfileDesignTests(unittest.TestCase):
    def _write_run(self, root, run_id, loss, x_m, y_m):
        path = root / "runs" / run_id / "run_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = path.parent / "best_model_state.pt"
        checkpoint_path.write_bytes(f"checkpoint::{run_id}".encode("utf-8"))
        path.write_text(
            json.dumps(
                {
                    "runtime": {"run_id": run_id, "source_position_mode": "single"},
                    "source": {"x_m": x_m, "y_m": y_m},
                    "checkpoint": {
                        "best_raw_loss": loss,
                        "path": str(checkpoint_path),
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_centers_on_minimum_loss_without_truth_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment_plan.json").write_text(
                json.dumps(
                    {
                        "source_domain_m": {
                            "x_min_m": -1000,
                            "x_max_m": 1000,
                            "y_min_m": -1000,
                            "y_max_m": 1000,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_run(root, "bad", 5.0, -300.0, 100.0)
            self._write_run(root, "best", 2.0, 100.0, 200.0)
            payload = generate_design(root, [250.0], [0], False)
            self.assertEqual(payload["provenance"]["selected_start_id"], "best")
            self.assertTrue(
                payload["provenance"]["selected_checkpoint_path"].endswith(
                    "best_model_state.pt"
                )
            )
            center = payload["resolved_candidates_m"][0]
            self.assertEqual((center["x_m"], center["y_m"]), (100.0, 200.0))
            self.assertEqual(len(payload["starts"]), 9)
            self.assertIn("does not read scenario truth", payload["truth_isolation"])

    def test_clips_and_deduplicates_boundary_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment_plan.json").write_text(
                json.dumps(
                    {
                        "source_domain_m": {
                            "x_min_m": 0,
                            "x_max_m": 100,
                            "y_min_m": 0,
                            "y_max_m": 100,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_run(root, "best", 1.0, 100.0, 100.0)
            payload = generate_design(root, [250.0, 500.0], [0, 1], True)
            coordinates = {
                (row["x_m"], row["y_m"])
                for row in payload["resolved_candidates_m"]
            }
            self.assertEqual(len(coordinates), len(payload["resolved_candidates_m"]))
            self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 100 for x, y in coordinates))
            self.assertEqual(len(payload["starts"]), 2 * len(coordinates))


if __name__ == "__main__":
    unittest.main()
