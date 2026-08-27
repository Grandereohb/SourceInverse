import csv
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_observed_response_proxy.py"
SPEC = importlib.util.spec_from_file_location("compare_observed_response_proxy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ObservedResponseProxyTests(unittest.TestCase):
    def test_fisher_exact_known_table(self):
        value = MODULE._fisher_exact_two_sided(4, 6, 22, 4)
        self.assertAlmostEqual(value, 0.013637156753691466, places=12)

    def test_observed_response_count(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concentration.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["time", "A", "B", "C", "TARGET_POLLUTANT"],
                )
                writer.writeheader()
                for index, values in enumerate(
                    ((1, 1, 1), (1, 2, 1), (1, 11, 1), (1, 1, 1))
                ):
                    writer.writerow(
                        {
                            "time": index,
                            "A": values[0],
                            "B": values[1],
                            "C": values[2],
                            "TARGET_POLLUTANT": "x",
                        }
                    )
            count, peak = MODULE._observed_response_count(path, 0.05)
            self.assertEqual(count, 1)
            self.assertEqual(peak, 10.0)


if __name__ == "__main__":
    unittest.main()
