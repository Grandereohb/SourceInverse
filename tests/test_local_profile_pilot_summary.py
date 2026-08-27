import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_local_profile_pilot import summarize_pilot  # noqa: E402


class LocalProfilePilotSummaryTests(unittest.TestCase):
    def _write_profile(self, root, scenario_id, center, ring_rises):
        path = root / scenario_id / "fixed_source_profile_candidates.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["candidate_id", "best_raw_loss"]
            )
            writer.writeheader()
            writer.writerow({"candidate_id": "selected", "best_raw_loss": center})
            for radius, rises in ring_rises.items():
                for index, rise in enumerate(rises):
                    writer.writerow(
                        {
                            "candidate_id": f"r{radius:04d}_d{index}",
                            "best_raw_loss": center * (1 + rise),
                        }
                    )
        return path

    def test_false_reassurance_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "scenario_selection": [
                            {"scenario_id": "success", "mechanism": "selected_success"},
                            {"scenario_id": "failure", "mechanism": "selection_failure_reachable"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sharp = {250: [0.10] * 8, 500: [0.20] * 8}
            self._write_profile(root, "success", 10.0, sharp)
            self._write_profile(root, "failure", 10.0, sharp)
            payload = summarize_pilot(config, root)
            self.assertFalse(payload["promotion_pass"])
            self.assertEqual(
                payload["false_reassurance_failure_scenarios"], ["failure"]
            )


if __name__ == "__main__":
    unittest.main()
