import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import contextlib
import io

import numpy as np

SPEC = importlib.util.spec_from_file_location('formal_analysis',
    Path(__file__).resolve().parents[1] / 'analyze_paper_formal.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_information_projection_and_ellipse():
    rng = np.random.default_rng(123)
    j = rng.normal(size=(30, 7))
    nuisance = list(range(2, 7))
    residual = j[:, :2] - j[:, 2:] @ np.linalg.lstsq(j[:, 2:], j[:, :2], rcond=None)[0]
    projected = analysis.source_information(j.T @ j, nuisance)
    np.testing.assert_allclose(projected, residual.T @ residual, atol=1e-12)
    diagnostic = analysis.ellipse_diagnostic(np.eye(2), .25, np.array([.5, 0]), 1000)
    assert diagnostic['available'] and diagnostic['contains_truth']
    assert diagnostic['mahalanobis_squared'] == 1
    assert not analysis.ellipse_diagnostic(np.diag([1., 0]), 1., np.zeros(2), 1000)['available']


def test_nuisance_information_cannot_improve_precision():
    rng = np.random.default_rng(100)
    j = rng.normal(size=(50, 9))
    full = j.T @ j
    q = analysis.source_information(full, list(range(4, 9)))
    qd = analysis.source_information(full, list(range(2, 9)))
    assert np.linalg.eigvalsh(full[:2, :2] - q).min() >= -1e-12
    assert np.linalg.eigvalsh(q - qd).min() >= -1e-12


class FormalAnalysisTests(unittest.TestCase):
    def test_projection_and_ellipse(self):
        test_information_projection_and_ellipse()

    def test_information_order(self):
        test_nuisance_information_cannot_improve_precision()

    def test_paired_bootstrap_preserves_pairing_and_requires_replication(self):
        rows = [dict(case_id=f'toy_{i}', cohort='toy', network='toy', wind='steady',
            release='constant', method=method, localization_error_m=float(i+offset),
            q_relative_l2=float(i+offset)/10)
            for i in range(8) for method, offset in
            (('ia_vpdi', 0), ('matched_joint_adam', 2), ('independent_puff_demo', 1))]
        effects = analysis.paired_effects(analysis.pd.DataFrame(rows), repeats=100)
        selected = effects[(effects.metric == 'localization_error_m') &
                           (effects.comparator == 'matched_joint_adam')].iloc[0]
        self.assertEqual(selected.IA_minus_comparator_median, -2.)
        self.assertEqual(selected.interval_low, -2.)
        self.assertEqual(selected.interval_high, -2.)
        sparse = analysis.paired_effects(analysis.pd.DataFrame(rows[:3]), repeats=100)
        self.assertTrue(sparse.interval_low.isna().all())
        shared = analysis.pd.DataFrame(rows).replace({'method': {
            'independent_puff_demo': 'shared_response_nnls_powell'}})
        self.assertIn('shared_response_nnls_powell',
                      set(analysis.paired_effects(shared, repeats=100).comparator))

    def test_partial_run_retains_pending_cases_and_checks_input_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [dict(case_id=f'toy_{i}', seed=i, cohort='nominal', network='toy',
                wind='steady', release='constant', repeat=i) for i in range(2)]
            (root/'protocol.json').write_text(json.dumps(dict(case_design=specs)))
            for spec in specs:
                case = root/spec['case_id']
                case.mkdir()
                (case/'input.json').write_text(json.dumps(spec))
                np.savez(case/'input.npz', observed=np.zeros(12),
                    clean=np.tile([1., .2], 6), stations=np.array([[0., 0.], [1., 0.]]))
            metadata = dict(specs[0], source_norm=[0., 0.], length_m=1.,
                station_peak_ratio_second_to_first=.2, noise_std=.01)
            method = dict(localization_error_m=0., q_relative_l2=0.,
                concentration_rmse_clean=0., observed_mse=.1, seconds=1.,
                source_at_boundary=False, source_norm=[0., 0.])
            result = dict(case_id='toy_0', metadata=metadata,
                methods={name: method for name in ('ia_vpdi', 'matched_joint_adam', 'independent_puff_demo')},
                conditional_information=dict(full_information=np.eye(5).tolist(),
                    profiled_theta_information=np.eye(4).tolist(),
                    profiled_source_information=np.eye(2).tolist(), sigma2=.1))
            case = root/'toy_0'
            (case/'result.json').write_text(json.dumps(result))
            np.savez(case/'optimization_traces.npz', candidate_surface=np.array([[0, 0, 1, 1, .1, 1]]))
            with contextlib.redirect_stdout(io.StringIO()):
                frame, _ = analysis.analyze(root)
            report = json.loads((root/'analysis/analysis_status.json').read_text())
            self.assertFalse(report['final'])
            self.assertEqual(report['completed_cases'], 1)
            self.assertEqual(len(frame), 3)
            signal = analysis.pd.read_csv(root/'analysis/case_characteristics.csv')
            self.assertEqual(len(signal), 1)
            self.assertAlmostEqual(signal.iloc[0].peak_ratio_second_to_first, .2)
            self.assertEqual(signal.iloc[0].stations_with_clean_peak_above_3sigma, 2)
            self.assertEqual(len(analysis.pd.read_csv(root/'analysis/case_status.csv')), 2)
            result['methods']['shared_response_nnls_powell'] = result['methods'].pop('independent_puff_demo')
            (case/'result.json').write_text(json.dumps(result))
            with contextlib.redirect_stdout(io.StringIO()):
                analysis.analyze(root)
            paired = analysis.pd.read_csv(root/'analysis/paired_localization.csv')
            self.assertIn('IA_minus_shared_powell_m', paired.columns)
            self.assertNotIn('IA_minus_puff_m', paired.columns)
            (case/'input.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                analysis.analyze(root)


if __name__ == '__main__':
    unittest.main()
