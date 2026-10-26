from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_paper_beam_comparison import method_rows, METRICS
from audit_paper_newman_result import chain_audit, MH_KEYS


class PublicAnalysisTests(unittest.TestCase):
    def record(self):
        return dict(event='source2_rel06', height_m=.3, complete=True, methods={
            'standard_gbest_pso': dict(replicates=[dict(
                {key: value for key in METRICS}, seed=seed)
                for seed, value in [(1, 2.), (2, 10.), (3, 5.)]])})

    def test_seed_median_and_absent_mass_reference(self):
        rows, raw = method_rows(self.record(), {'source2_rel06': dict(
            reference_q_kg_h=None, q_reference_status='missing')})
        self.assertEqual(rows[0]['localization_error_m'], 5.)
        self.assertEqual(rows[0]['seconds_all_replicates'], 17.)
        self.assertEqual(len(raw), 3)
        self.assertIsNone(rows[0]['q_relative_error'])

    def test_nonfinite_seed_not_silently_discarded(self):
        record = self.record()
        record['methods']['standard_gbest_pso']['replicates'][1]['seconds'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            method_rows(record, {'source2_rel06': dict(reference_q_kg_h=None)})

    def test_mh_changes_exclude_gibbs_and_burn(self):
        arrays = {key: np.array([1., 1., 2., 2., 3., 3.]) for key in MH_KEYS}
        arrays['background'] = np.arange(6.)
        report = chain_audit(arrays, 2)
        self.assertEqual(report['source_x']['distinct_retained_states'], 2)
        self.assertAlmostEqual(report['mh_block']['retained_transition_fraction'], 1/3)
        self.assertAlmostEqual(report['mh_block']['all_draw_transition_fraction'], 2/5)

    def test_reject_nonfinite_mh_chain(self):
        arrays = {key: np.ones(6) for key in MH_KEYS}
        arrays['source_x'][3] = np.inf
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            chain_audit(arrays, 2)


if __name__ == '__main__':
    unittest.main()
