import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pinn_source'))
sys.path.insert(0, str(ROOT/'scripts'))
import field
import test_substep_diagnostics as fixtures
from inversion_validation_common import configured_variant, fixed_q_transport_control, exact_release_decay_mass


class SourceDecayIntegrationTests(unittest.TestCase):
    def test_positive_weights_match_quadrature_including_zero_and_large_decay(self):
        for z in (0., 1e-10, 1e-4, .001, .01, .5, 5., 100., 1000.):
            attenuation, left, right = field.source_decay_weights(1., z)
            self.assertGreaterEqual(min(attenuation, left, right), 0.)
            self.assertTrue(np.isfinite([attenuation, left, right]).all())
            expected_left = quad(lambda t: (1-t)*np.exp(-z*(1-t)), 0., 1., epsabs=1e-12)[0]
            expected_right = quad(lambda t: t*np.exp(-z*(1-t)), 0., 1., epsabs=1e-12)[0]
            np.testing.assert_allclose([left, right], [expected_left, expected_right], rtol=1e-8, atol=1e-12)
        for dt, decay in ((0., 1.), (1., -1.), (float('nan'), 1.)):
            with self.assertRaises(ValueError):
                field.source_decay_weights(dt, decay)

    def test_linear_release_decay_mass_exact_at_every_output_time(self):
        prototype = fixtures.SubstepDiagnosticTests().prototype()
        prototype.recurrent_source_decay_integrator = 'exponential_linear'
        knots = prototype.recurrent_times.cpu().numpy()
        coefficients = np.array([.1, 2., .4, 0.])
        query = np.array([[0., 0., t] for t in knots])
        for steps in (1, 2, 4):
            model = configured_variant(prototype, 21, substeps=steps, preserve_diffusion_mode=True)
            self.assertEqual(model.recurrent_source_decay_integrator, 'exponential_linear')
            _, moments, trace = fixed_q_transport_control(model, .1, query, knots, coefficients, 'transport')
            np.testing.assert_allclose([r['mass_norm'] for r in moments],
                exact_release_decay_mass(model, knots, coefficients), rtol=3e-7, atol=1e-8)
            self.assertTrue(all(abs(r['injection_balance_error']) < 1e-7 for r in trace))

    def check_linearity(self, device):
        model = fixtures.SubstepDiagnosticTests().prototype(device)
        model.recurrent_source_decay_integrator = 'exponential_linear'
        q1 = torch.linspace(0., 1., model.recurrent_integration_times.numel(), device=device)
        q2 = q1.flip(0)*.3
        f1 = field.recurrent_plume_fields_from_q(model, .1, q1)
        f2 = field.recurrent_plume_fields_from_q(model, .1, q2)
        both = field.recurrent_plume_fields_from_q(model, .1, q1+q2)
        torch.testing.assert_close(both, f1+f2, rtol=2e-6, atol=1e-7)
        self.assertTrue(bool(torch.isfinite(both).all() & (both >= 0).all()))
        self.assertEqual(both.device, q1.device)
        q = q1.clone().requires_grad_()
        field.recurrent_plume_fields_from_q(model, .1, q).sum().backward()
        self.assertTrue(bool(torch.isfinite(q.grad).all()))

    def test_cpu_linearity_positivity_and_gradient(self):
        self.check_linearity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_linearity_positivity_and_gradient(self):
        self.check_linearity('cuda:0')

    def test_zero_decay_matches_legacy_and_default_is_legacy(self):
        prototype = fixtures.SubstepDiagnosticTests().prototype()
        prototype.recurrent_decay_norm = 0.
        legacy = configured_variant(prototype, 21, substeps=1, preserve_diffusion_mode=True)
        self.assertEqual(legacy.recurrent_source_decay_integrator, 'legacy')
        updated = configured_variant(legacy, 21, substeps=1, preserve_diffusion_mode=True)
        updated.recurrent_source_decay_integrator = 'exponential_linear'
        q = torch.ones_like(legacy.recurrent_integration_times)
        torch.testing.assert_close(field.recurrent_plume_fields_from_q(legacy, .1, q),
            field.recurrent_plume_fields_from_q(updated, .1, q), rtol=0., atol=0.)

    def test_output_time_reconfiguration_preserves_mode_and_restores_context(self):
        model = fixtures.SubstepDiagnosticTests().prototype()
        model.recurrent_source_decay_integrator = 'exponential_linear'
        times = model.recurrent_times.clone()
        plans = model.recurrent_source_decay_plans
        original_q = model.Q
        model.Q = lambda t: torch.ones_like(t)
        try:
            field.recurrent_plume_fields_at_times(model, .1, times, model.recurrent_u, model.recurrent_v)
            self.assertEqual(model.recurrent_source_decay_integrator, 'exponential_linear')
            self.assertIs(model.recurrent_source_decay_plans, plans)
            torch.testing.assert_close(model.recurrent_times, times)
        finally:
            model.Q = original_q


if __name__ == '__main__':
    unittest.main()
