import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from simple_baselines import maximum_anomaly_station, maximum_anomaly_upwind


class SimpleBaselineTests(unittest.TestCase):
    def _write_inputs(self, root: Path):
        sites = root / "sites.csv"
        concentration = root / "concentration.csv"
        wind = root / "wind.csv"
        pd.DataFrame(
            {"station": ["west", "east"], "lon": [121.0, 121.01], "lat": [31.0, 31.0]}
        ).to_csv(sites, index=False)
        pd.DataFrame(
            {
                "time": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
                "west": [1.0, 1.0],
                "east": [1.0, 10.0],
            }
        ).to_csv(concentration, index=False)
        pd.DataFrame(
            {
                "time": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
                "dir": [270.0, 270.0],
                "sp": [1.0, 1.0],
            }
        ).to_csv(wind, index=False)
        return sites, concentration, wind

    def test_b0_selects_maximum_positive_station_anomaly(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp))
            result = maximum_anomaly_station(*paths)
        self.assertEqual(result["peak_station"], "east")
        self.assertEqual(result["peak_anomaly"], 4.5)

    def test_b1_moves_against_downwind_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp))
            b0 = maximum_anomaly_station(*paths)
            b1 = maximum_anomaly_upwind(
                *paths, distance_m=100.0, source_position_pad_m=500.0
            )
        self.assertLess(b1["source_x_m"], b0["source_x_m"])
        self.assertAlmostEqual(b1["source_y_m"], b0["source_y_m"], places=6)

    def test_b1_rejects_negative_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp))
            with self.assertRaisesRegex(ValueError, "non-negative"):
                maximum_anomaly_upwind(
                    *paths, distance_m=-1.0, source_position_pad_m=500.0
                )


if __name__ == "__main__":
    unittest.main()
