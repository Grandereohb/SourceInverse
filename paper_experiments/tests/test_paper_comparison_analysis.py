from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_paper_comparison_v4 import METRICS, method_rows, ellipse_size


class ComparisonAnalysisTests(unittest.TestCase):
    def test_ellipse_area_scales_as_square_length(self):
        small = ellipse_size(np.diag([4., 1.]), 2., 10., True)
        large = ellipse_size(np.diag([4., 1.]), 2., 20., True)
        self.assertAlmostEqual(large['area_m2']/small['area_m2'], 4.)
        self.assertAlmostEqual(small['area_m2'], np.pi*5.991*100.)
        self.assertIsNone(ellipse_size(None, None, None, False)['area_m2'])

    def record(self):
        runs = [dict({key: value for key in METRICS}, source_at_boundary=False, seed=seed)
                for seed, value in [(1729, 1.), (1730, 100.), (1731, 4.)]]
        return dict(metadata=dict(case_id='unit', cohort='nominal', network='ideal5',
                    wind='steady', release='pulse', repeat=0),
                    methods={'standard_gbest_pso': dict(replicates=runs)})

    def test_pso_median_is_within_case_not_best_seed(self):
        rows, raw = method_rows(self.record())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(raw), 3)
        self.assertEqual(rows[0]['localization_error_m'], 4.)
        self.assertEqual(rows[0]['seconds_all_replicates'], 105.)
        self.assertEqual(rows[0]['replicates'], 3)

    def test_nonfinite_replicate_is_not_silently_removed(self):
        record = self.record()
        record['methods']['standard_gbest_pso']['replicates'][1]['q_relative_l2'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            method_rows(record)


if __name__ == '__main__':
    unittest.main()
