import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from summarize_fixed_source_profile import summarize_profile  # noqa: E402


class FixedSourceProfileTests(unittest.TestCase):
    def _write_run(
        self, root, candidate_id, x_m, loss, mode="fixed", seed=0, run_id=None
    ):
        run_id = run_id or candidate_id
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        payload = {
            "runtime": {
                "source_position_mode": mode,
                "source_init_override_m": [x_m, 2.0],
                "recurrent_solver": "production",
                "run_id": run_id,
                "completed_epochs": 10,
                "training_wall_time_s": 1.5,
                "q_mode": "constant",
            },
            "source": {"x_m": x_m, "y_m": 2.0},
            "checkpoint": {"best_raw_loss": loss, "best_epoch": 7},
            "random_seed": seed,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_delta_profile_loss_is_relative_to_reoptimized_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "left", 1.0, 5.0)
            self._write_run(root, "right", 3.0, 8.5)
            payload = summarize_profile(root)
        rows = {row["candidate_id"]: row for row in payload["rows"]}
        self.assertEqual(rows["left"]["delta_profile_loss"], 0.0)
        self.assertEqual(rows["right"]["delta_profile_loss"], 3.5)
        self.assertIn("not a posterior probability", payload["interpretation"])

    def test_multiple_nuisance_starts_are_minimized_within_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "resolved_starts": [
                    {
                        "id": "left_0",
                        "candidate_id": "left",
                        "nuisance_start_id": "seed0",
                    },
                    {
                        "id": "left_1",
                        "candidate_id": "left",
                        "nuisance_start_id": "seed1",
                    },
                    {
                        "id": "right_0",
                        "candidate_id": "right",
                        "nuisance_start_id": "seed0",
                    },
                ]
            }
            (root / "experiment_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            self._write_run(root, "left", 1.0, 9.0, seed=0, run_id="left_0")
            self._write_run(root, "left", 1.0, 4.0, seed=1, run_id="left_1")
            self._write_run(root, "right", 3.0, 6.0, seed=0, run_id="right_0")
            payload = summarize_profile(root)
        rows = {row["candidate_id"]: row for row in payload["rows"]}
        self.assertEqual(rows["left"]["best_raw_loss"], 4.0)
        self.assertEqual(rows["left"]["nuisance_start_id"], "seed1")
        self.assertEqual(rows["left"]["nuisance_run_count"], 2)
        self.assertEqual(rows["right"]["delta_profile_loss"], 2.0)

    def test_rejects_movable_source_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "bad", 1.0, 5.0, mode="single")
            with self.assertRaisesRegex(ValueError, "non-fixed"):
                summarize_profile(root)


if __name__ == "__main__":
    unittest.main()
