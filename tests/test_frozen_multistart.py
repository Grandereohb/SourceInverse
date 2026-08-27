import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_frozen_multistart.py"
SPEC = importlib.util.spec_from_file_location("run_frozen_multistart", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FrozenMultistartTests(unittest.TestCase):
    def test_load_design_and_resolve_starts_without_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            design_path = Path(tmp) / "design.json"
            design_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "starts": [
                            {"id": "heuristic", "mode": "pipeline_heuristic"},
                            {
                                "id": "corner",
                                "mode": "domain_fraction",
                                "fraction_x": 0.25,
                                "fraction_y": 0.75,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            design = MODULE.load_design(design_path)
            resolved = MODULE.resolve_starts(
                design,
                {
                    "x_min_m": -100.0,
                    "x_max_m": 300.0,
                    "y_min_m": -200.0,
                    "y_max_m": 200.0,
                },
            )
            self.assertIsNone(resolved[0]["source_init_override_m"])
            self.assertEqual(resolved[1]["source_init_override_m"], [0.0, 100.0])

    def test_source_domain_bounds_follow_station_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            site_path = Path(tmp) / "sites.csv"
            pd.DataFrame(
                {
                    "station": ["a", "b", "c"],
                    "lon": [121.0, 121.01, 121.02],
                    "lat": [31.0, 31.01, 31.02],
                }
            ).to_csv(site_path, index=False)
            with patch.object(MODULE.runtime_config, "SOURCE_POSITION_PAD_M", 250.0):
                sites, _, _ = MODULE.load_sites(site_path)
                bounds = MODULE.source_domain_bounds(site_path)
            self.assertEqual(bounds["x_min_m"], sites["x"].min() - 250.0)
            self.assertEqual(bounds["x_max_m"], sites["x"].max() + 250.0)
            self.assertEqual(bounds["y_min_m"], sites["y"].min() - 250.0)
            self.assertEqual(bounds["y_max_m"], sites["y"].max() + 250.0)

    def test_design_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "starts": [
                            {"id": "same", "mode": "pipeline_heuristic"},
                            {"id": "same", "mode": "pipeline_heuristic"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                MODULE.load_design(path)

    def test_code_snapshot_detects_relevant_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = "module.py"
            path = root / relative
            path.write_text("version = 1\n", encoding="utf-8")
            with patch.object(MODULE, "CODE_SNAPSHOT_RELATIVE_PATHS", (relative,)):
                snapshot = MODULE._code_snapshot(root)
                MODULE._verify_code_snapshot(snapshot, root)
                path.write_text("version = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "module.py"):
                    MODULE._verify_code_snapshot(snapshot, root)


if __name__ == "__main__":
    unittest.main()
