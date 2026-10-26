import sys
from pathlib import Path
import unittest
from collections import Counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_paper_formal_experiments as formal


class FormalProtocolTests(unittest.TestCase):
    def test_interval_mean_wind_preserves_exact_total_displacement(self):
        model,_,_=formal.parcel.configure({'wind':'veering'},[[0.,0.]],
                                         wind_interval_mean=True)
        duration=model.recurrent_times[1:]-model.recurrent_times[:-1]
        actual=torch.stack(((model.recurrent_u[:-1]*duration).sum(),
                            (model.recurrent_v[:-1]*duration).sum())).numpy()
        expected=1.2/.15*np.array([np.sin(.55)-np.sin(-.35),np.cos(-.35)-np.cos(.55)])
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)

    def test_design_counts_unique_seeds_and_marginal_balance(self):
        cases=formal.design()
        self.assertEqual(Counter(c['cohort'] for c in cases),
                         Counter(nominal=96,wind_mismatch=12,q_mismatch=12))
        self.assertEqual(len({c['seed'] for c in cases}),120)
        self.assertEqual(len({c['case_id'] for c in cases}),120)
        for cohort in ('nominal','wind_mismatch','q_mismatch'):
            selected=[c for c in cases if c['cohort']==cohort]
            self.assertEqual(Counter(c['network'] for c in selected)['ideal5'],len(selected)//2)
            self.assertEqual(Counter(c['wind'] for c in selected)['steady'],len(selected)//2)
        self.assertFalse(any(c['release']=='constant' for c in cases if c['cohort']=='q_mismatch'))

    def test_generator_is_reproducible_and_does_not_clip_noise(self):
        spec=dict(case_id='unit_case',seed=995001,cohort='nominal',network='ideal5',
                  wind='steady',release='pulse',repeat=0)
        network=dict(stations=[[.3,-.1],[.5,1.],[1.,-1.],[1.7,.9],[2.,-.9]],length_m=1000.,sigma=.15)
        a,x=formal.make_input(spec,network)
        b,y=formal.make_input(spec,network)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(x['observed'],y['observed'])
        self.assertEqual(x['observed'].shape,(30,))
        self.assertAlmostEqual(float(x['clean'].max()),1.)
        self.assertLess(a['release_quadrature_relative_change'],1e-4)
        self.assertLess(a['temporal_quadrature_relative_change'],1e-3)
        self.assertTrue((x['observed']<0).any())


if __name__=='__main__':
    unittest.main()
