import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pinn_source'))
from background import estimate_common_background, nonnegative_background_floor
import pipeline


class BackgroundTests(unittest.TestCase):
    def test_diagnostic_reader_supports_consolidated_and_historical_reports(self):
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1]/'scripts'), *sys.path]):
            from inversion_validation_common import load_ia_vpdi_report
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            legacy = folder/'ia_vpdi_report.json'
            legacy.write_text(json.dumps({'mode': 'ia_vpdi', 'best_profiled_score': 2.}), encoding='utf-8')
            self.assertEqual(load_ia_vpdi_report(folder)['best_profiled_score'], 2.)
            consolidated = folder/'result_quality_report.json'
            consolidated.write_text(json.dumps({'ia_vpdi': {'mode': 'ia_vpdi', 'best_profiled_score': 1.}}), encoding='utf-8')
            self.assertEqual(load_ia_vpdi_report(folder)['best_profiled_score'], 1.)
            legacy.unlink()
            self.assertEqual(load_ia_vpdi_report(folder)['best_profiled_score'], 1.)

    def test_config_has_one_algorithm_selector_and_rejects_invalid_mode(self):
        import config
        self.assertFalse(hasattr(config, 'MODEL_NAME'))
        self.assertIn(config.INVERSION_MODE, ('joint_adam', 'ia_vpdi'))
        env = dict(os.environ)
        env.pop('PINN_INVERSION_MODE', None)
        with patch.dict(os.environ, env, clear=True), patch.object(pipeline, 'INVERSION_MODE', 'unknown'):
            with self.assertRaisesRegex(ValueError, 'INVERSION_MODE'):
                pipeline.run('unused', 'unused', 'unused', random_seed=None)

    def test_existing_estimators_unchanged_and_disabled_index_preserved(self):
        frame = pd.DataFrame([[0., 1., 4.], [0., 3., 6.]], index=[2, 5])
        for mode, expected in [('median', frame.median(axis=1)),
                               ('q25', frame.quantile(.25, axis=1)),
                               ('q40', frame.quantile(.40, axis=1))]:
            pd.testing.assert_series_equal(estimate_common_background(frame, True, mode), expected)
        pd.testing.assert_series_equal(estimate_common_background(frame, False),
                                       pd.Series([0., 0.], index=frame.index))
        with self.assertRaises(ValueError):
            estimate_common_background(frame, True, 'typo')

    def test_error_floor_for_zero_station_and_nonnegative_predictions(self):
        y = [0., 4., 0., 8.]
        b = [2., 2., 3., 3.]
        report = nonnegative_background_floor(y, b, ['low', 'high'] * 2, [2., 4., 3., 8.])
        self.assertEqual(report['raw_sse_lower_bound'], 13.)
        self.assertEqual(report['fraction_of_current_raw_sse'], 1.)
        self.assertAlmostEqual(report['raw_rmse_lower_bound'], np.sqrt(13./4))
        self.assertEqual(report['observations_below_background'], 2)
        self.assertEqual(report['station_raw_rmse_lower_bounds']['high'], 0.)
        self.assertTrue(report['nonnegative_plume_assumption_satisfied'])
        raw = nonnegative_background_floor(y, np.zeros(4), ['low', 'high'] * 2, y)
        self.assertEqual(raw['raw_sse_lower_bound'], 0.)
        self.assertIsNone(raw['fraction_of_current_raw_sse'])
        not_applicable = nonnegative_background_floor(y, b, ['low', 'high'] * 2, y)
        self.assertFalse(not_applicable['nonnegative_plume_assumption_satisfied'])
        self.assertIsNone(not_applicable['fraction_of_current_raw_sse'])

    def test_floor_validates_arrays(self):
        for values in (([], [], [], []), ([1.], [1., 2.], ['a'], [1.]),
                       ([np.nan], [0.], ['a'], [0.])):
            with self.assertRaises(ValueError):
                nonnegative_background_floor(*values)

    def test_raw_pipeline_smoke_in_both_modes(self):
        self.check_raw_pipeline("grid")

    def test_raw_parcel_pipeline_smoke_in_both_modes(self):
        self.check_raw_pipeline("gaussian_parcels", configured_mode=True)

    def check_raw_pipeline(self, transport, configured_mode=False):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            times = pd.date_range('2026-01-01', periods=4, freq='h')
            sites = folder/'sites.xlsx'
            concentration = folder/'concentration.xlsx'
            wind = folder/'wind.xlsx'
            pd.DataFrame({'station': ['low', 'weak', 'peak'], 'lon': [117., 117.01, 117.02],
                          'lat': [39., 39.01, 39.02]}).to_excel(sites, index=False)
            pd.DataFrame({'time': times, 'low': [0., 0., 0., 0.], 'weak': [1., 2., 3., 4.],
                          'peak': [2., 5., 2., 1.]}).to_excel(concentration, index=False)
            pd.DataFrame({'time': times, 'dir': [0., 30., 60., 90.], 'sp': [.5]*4}).to_excel(wind, index=False)
            for mode in ('joint_adam', 'ia_vpdi'):
                env = dict(os.environ)
                env.update({'PINN_TRAIN_ON_RESIDUAL': '0', 'PINN_INVERSION_MODE': mode,
                       'PINN_DEVICE': 'cpu', 'PINN_EPOCHS': '1', 'PINN_IA_VP_TOP_K': '1',
                       'PINN_IA_VP_OUTER_STEPS': '1', 'PINN_IA_VP_Q_LAYOUT_STRATEGY': 'uniform'})
                if configured_mode:
                    env.pop('PINN_INVERSION_MODE', None)
                env.update(PINN_TRANSPORT_SOLVER=transport, PINN_PARCEL_SUBDIVISIONS='16', PINN_SIGMA_SRC_M='500')
                with (patch.dict(os.environ, env, clear=True), patch.multiple(pipeline,
                    ENABLE_EVENT_WINDOW_CROP=False, USE_SOURCE_LANDSCAPE_CONFIDENCE=False,
                    INVERSION_MODE=mode if configured_mode else 'joint_adam',
                    IA_VP_DIFFUSION_SCALE_SEEDS=(1.,), IA_VP_DIFFUSION_RATIO_SEEDS=(1.,)),
                    contextlib.redirect_stdout(io.StringIO()) as console):
                    result = pipeline.run(sites, concentration, wind, output_dir=folder/mode,
                                          result_root_dir=folder/'exports', make_plots=False)
                self.assertFalse(result['train_on_residual'])
                self.assertEqual(result['inversion_mode'], mode)
                self.assertEqual(result['baseline_mode'], 'none')
                report = json.loads((Path(result['output_dir'])/'result_quality_report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['model']['baseline_mode'], 'none')
                case_dir = Path(result['output_dir'])
                self.assertEqual(list(case_dir.glob('ia_vpdi_*.csv')), [])
                self.assertFalse((case_dir/'ia_vpdi_report.json').exists())
                if mode == 'ia_vpdi':
                    self.assertEqual(report['ia_vpdi']['mode'], mode)
                    self.assertIn('q_coefficients', report['ia_vpdi'])
                    self.assertIn('profile_information', report['ia_vpdi'])
                    with pd.ExcelFile(case_dir/'diagnostics.xlsx') as workbook:
                        self.assertIn('ia_vpdi_candidates', workbook.sheet_names)
                        candidates = pd.read_excel(workbook, sheet_name='ia_vpdi_candidates')
                        self.assertGreater(len(candidates), 0)
                        self.assertIn('profile_score', candidates.columns)
                floor = report['model']['background_error_floor']
                self.assertEqual(floor['raw_rmse_lower_bound'], 0.)
                self.assertEqual(floor['observation_count'], 12)
                self.assertTrue(np.isfinite(result['total_loss']))
                self.assertEqual(report['recurrent_pde']['transport_solver'], transport)
                self.assertEqual(report['recurrent_pde']['transport_boundary'], 'full_space' if transport == 'gaussian_parcels' else 'absorbing_zero_exterior')
                self.assertEqual(report['recurrent_pde']['grid_substep_diagnostics_applied'], transport == 'grid')
                if transport == 'gaussian_parcels':
                    self.assertIn('grid_CFL=not_applicable', console.getvalue())
                    self.assertNotIn('Warning: adaptive recurrent substeps', console.getvalue())
                    self.assertIn('diffusion_solver=analytic_tensor_covariance', console.getvalue())


if __name__ == '__main__':
    unittest.main()
