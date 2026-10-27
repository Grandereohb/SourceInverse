from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paper_uncertainty_calibration as calibration
import run_paper_manuscript_supplement as supplement


class CalibrationTests(unittest.TestCase):
    def test_finite_sample_rank_and_insufficient_size(self):
        self.assertEqual(calibration.calibration_threshold(np.arange(48.)),46.)
        self.assertTrue(np.isinf(calibration.calibration_threshold(np.arange(10.))))
        with self.assertRaises(ValueError):
            calibration.calibration_threshold([float('nan')])

    def test_validation_truth_cannot_change_scale(self):
        rows=[dict(case_id=f'case_{i}',cohort='nominal',diagnostic='eliminate_Q_D',
            network='five' if i%2 else 'six',repeat=i//12,available=True,
            mahalanobis_squared=float(i+1),major_m=10.) for i in range(96)]
        data=pd.DataFrame(rows)
        cases,_,first=calibration.calibrate(data)
        data.loc[data['repeat']>=4,'mahalanobis_squared']*=1000
        _,_,second=calibration.calibrate(data)
        self.assertEqual(first,second)
        a=set(cases.loc[cases.partition=='calibration','case_id'])
        b=set(cases.loc[cases.partition=='validation','case_id'])
        self.assertFalse(a&b)
        self.assertEqual(len(a|b),96)

    def test_scenario_geometry_is_distinct_and_deterministic(self):
        scenarios=supplement.scenarios()
        self.assertEqual(scenarios,supplement.scenarios())
        self.assertEqual(len(scenarios['surround5_pulse']['stations']),5)
        self.assertEqual(len(scenarios['corridor6_double_peak']['stations']),6)

    def test_unavailable_case_keeps_fixed_denominator_and_infinite_score(self):
        rows=[dict(case_id=f'case_{i}',cohort='nominal',diagnostic='eliminate_Q_D',
            network='five' if i%2 else 'six',repeat=i//12,available=True,
            mahalanobis_squared=float(i+1),major_m=10.) for i in range(96)]
        rows[0].update(available=False, mahalanobis_squared=np.nan, major_m=np.nan)
        cases, summary, threshold = calibration.calibrate(pd.DataFrame(rows))
        self.assertEqual(len(cases), 96)
        self.assertEqual(threshold, 48.)
        self.assertTrue(np.isinf(cases.loc[0, 'calibration_score']))
        self.assertFalse(cases.loc[0, 'calibrated_contains_truth'])
        group = summary[(summary.partition == 'calibration') & (summary.network == 'all')].iloc[0]
        self.assertEqual(group.n, 48)
        self.assertEqual(group.defined_ellipses, 47)


if __name__=='__main__':
    unittest.main()
