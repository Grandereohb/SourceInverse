from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_paper_newman_batch import chain_summary


class NewmanAnalysisTests(unittest.TestCase):
    def test_stuck_chain_kept_with_undefined_ess_not_filled_with_zero(self):
        record = dict(effective_sample_size=dict(source_x=[np.nan], source_y=[np.nan], s=[1.]))
        audit = dict(chains=dict(mh_block=dict(retained_transition_fraction=0.),
                                source_x=dict(distinct_retained_states=1)))
        result = chain_summary(record, audit)
        self.assertIsNone(result['position_min_ess'])
        self.assertEqual(result['sampling_status'], 'stuck_retained_mh_chain')
        self.assertFalse(result['posterior_convergence_certified'])

    def test_finite_ess_does_not_certify_single_chain_convergence(self):
        record = dict(effective_sample_size=dict(source_x=[150.], source_y=[80.], s=[175.]))
        audit = dict(chains=dict(mh_block=dict(retained_transition_fraction=.58),
                                source_x=dict(distinct_retained_states=2901)))
        result = chain_summary(record, audit)
        self.assertEqual(result['position_min_ess'], 80.)
        self.assertEqual(result['sampling_status'], 'fixed_budget_single_chain')
        self.assertFalse(result['posterior_convergence_certified'])


if __name__ == '__main__':
    unittest.main()
