import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data_io import load_conc, load_wind  # noqa: E402
from pipeline import _copy_training_inputs  # noqa: E402


class TabularInputTests(unittest.TestCase):
    def test_concentration_and_wind_csv_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conc_path = root / "concentration.csv"
            wind_path = root / "wind.csv"
            pd.DataFrame(
                {
                    "time": ["2026-01-01 00:00:00"],
                    "station_a": [3.5],
                    "TARGET_POLLUTANT": ["synthetic"],
                }
            ).to_csv(conc_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                {
                    "time": ["2026-01-01 00:00:00"],
                    "dir": [90.0],
                    "sp": [2.0],
                }
            ).to_csv(wind_path, index=False, encoding="utf-8-sig")

            concentration = load_conc(conc_path)
            wind = load_wind(wind_path)
            self.assertEqual(concentration.attrs["target_pollutant"], "synthetic")
            self.assertEqual(float(concentration["station_a"].iloc[0]), 3.5)
            self.assertEqual(float(wind["sp"].iloc[0]), 2.0)

    def test_input_copy_preserves_actual_file_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            site = root / "sites.xlsx"
            concentration = root / "concentration.csv"
            wind = root / "wind.tsv"
            site.write_bytes(b"xlsx-placeholder")
            concentration.write_text("time,value\n", encoding="utf-8")
            wind.write_text("time\tdir\tsp\n", encoding="utf-8")

            copied = _copy_training_inputs(output, site, concentration, wind)
            self.assertEqual(
                set(copied), {"sites.xlsx", "concentration.csv", "wind.tsv"}
            )
            self.assertTrue((output / "concentration.csv").exists())
            self.assertTrue((output / "wind.tsv").exists())


if __name__ == "__main__":
    unittest.main()
