import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pinn_source"))
import field
from models.pinn import PINN
from inversion_validation_common import configured_variant, analytic_release_predictions, isolated_operator_moments, local_source_candidates, response_amplitude_diagnostics, trace_recurrent_mass, extend_boundary_candidates, summarize_near_equivalent_candidates
from variable_projection import build_q_basis, build_response_matrix


class SubstepDiagnosticTests(unittest.TestCase):
    def test_release_decay_control_converges_to_scalar_reference(self):
        from inversion_validation_common import fixed_q_transport_control, exact_release_decay_mass
        prototype = self.prototype()
        knots = prototype.recurrent_times.cpu().numpy()
        coefficients = np.ones(len(knots))
        query = np.array([[0., 0., t] for t in knots])
        errors = []
        for steps in (1, 2, 4):
            model = configured_variant(prototype, 21, substeps=steps, preserve_diffusion_mode=True)
            _, moments, trace = fixed_q_transport_control(model, .1, query, knots, coefficients,
                                                         disabled_operator='transport')
            exact = exact_release_decay_mass(model, knots, coefficients)
            expected = field.RECURRENT_SOURCE_SCALE*(field.RECURRENT_INITIAL_RELEASE_FRACTION
                *model.recurrent_initial_release_dt*np.exp(-model.recurrent_decay_norm*knots)
                +(1-np.exp(-model.recurrent_decay_norm*knots))/model.recurrent_decay_norm)
            np.testing.assert_allclose(exact, expected, rtol=1e-6)
            errors.append(np.max(np.abs(np.array([r['mass_norm'] for r in moments])-exact)))
            for row in trace:
                self.assertAlmostEqual(row['mass_after_advection'], row['mass_after_first_injection'], places=7)
                self.assertAlmostEqual(row['mass_after_diffusion'], row['mass_after_advection'], places=7)
        self.assertTrue(np.all(np.diff(errors) < 0))
        self.assertLess(errors[-1], errors[0]/10)

    def test_fixed_q_control_preserves_release_and_does_not_fit(self):
        from inversion_validation_common import fixed_q_transport_control
        model = self.prototype()
        knots = model.recurrent_times.cpu().numpy()
        coefficients = np.array([0., 1., .5, 0.])
        xyt = np.array([[0., 0., t] for t in knots])
        for steps in (1, 2, 4):
            variant = configured_variant(model, 21, substeps=steps, preserve_diffusion_mode=True)
            expected_q = np.interp(variant.recurrent_integration_times.cpu().numpy(), knots, coefficients)
            expected = field.recurrent_plume_value_from_q(variant,
                torch.as_tensor(xyt, dtype=variant.xs.dtype), .1, expected_q).detach().numpy().reshape(-1)
            pred, moments, trace = fixed_q_transport_control(variant, .1, xyt, knots, coefficients)
            np.testing.assert_allclose(pred, expected, atol=1e-7)
            self.assertEqual(len(moments), len(knots))
            self.assertEqual(len(trace), (len(knots)-1)*steps)
            self.assertTrue(all(r['mass_norm'] >= 0 for r in moments))
        np.testing.assert_array_equal(coefficients, [0., 1., .5, 0.])
        with self.assertRaises(ValueError):
            fixed_q_transport_control(model, .1, xyt, knots, -np.ones(4))

    def test_resolution_variants_hold_time_and_physics_fixed(self):
        model = self.prototype()
        for n in (36, 48, 64):
            variant = configured_variant(model, n, substeps=1, preserve_diffusion_mode=True)
            self.assertEqual(variant.recurrent_x_grid.numel(), n)
            self.assertEqual(variant.recurrent_y_grid.numel(), n)
            torch.testing.assert_close(variant.recurrent_integration_times, model.recurrent_integration_times)
            torch.testing.assert_close(variant.recurrent_u, model.recurrent_u)
            torch.testing.assert_close(variant.recurrent_v, model.recurrent_v)
            self.assertEqual(variant.recurrent_decay_norm, model.recurrent_decay_norm)
            self.assertEqual(variant.recurrent_initial_release_dt, model.recurrent_initial_release_dt)
            self.assertEqual(variant.recurrent_d_scale_norm, model.recurrent_d_scale_norm)
            torch.testing.assert_close(variant.recurrent_x_grid[[0, -1]], model.recurrent_x_grid[[0, -1]])
            torch.testing.assert_close(variant.recurrent_y_grid[[0, -1]], model.recurrent_y_grid[[0, -1]])

    def test_near_equivalent_sets_use_relative_mse_not_confidence(self):
        records = [dict(x_norm=x, y_norm=0., normalized_mse=score, q_max_model_units=q,
                        at_allowed_boundary=False, at_candidate_window_edge=x == 0.)
                   for x,score,q in ((0., 1., 10.), (.1, 1.04, 100.), (.2, 1.2, 1000.))]
        small, wide = summarize_near_equivalent_candidates(records, 10000., (.01, .05))
        self.assertEqual(small["candidate_count"], 1)
        self.assertEqual(wide["candidate_count"], 2)
        self.assertEqual(wide["sampled_diameter_m"], 1000.)
        self.assertEqual(wide["q_peak_ratio"], 10.)
        self.assertTrue(wide["touches_candidate_window_edge"])
        for invalid in (-1., float("nan")):
            with self.assertRaises(ValueError):
                summarize_near_equivalent_candidates(records, 10000., (invalid,))

    def test_mass_trace_preserves_original_forward_and_restores_operators(self):
        model = self.prototype()
        q = torch.ones_like(model.recurrent_integration_times)
        original = field.recurrent_plume_fields_from_q(model, .1, q)
        traced, rows = trace_recurrent_mass(model, .1, q)
        torch.testing.assert_close(traced, original)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(abs(row["injection_balance_error"]) < 1e-6 for row in rows))
        for control, stage, previous in (("advection", "mass_after_advection", "mass_after_first_injection"),
                ("diffusion", "mass_after_diffusion", "mass_after_advection")):
            _, controlled = trace_recurrent_mass(model, .1, q, control)
            for row in controlled:
                self.assertAlmostEqual(row[stage], row[previous], places=6)
        torch.testing.assert_close(field.recurrent_plume_fields_from_q(model, .1, q), original)

    def test_boundary_extension_preserves_candidates_and_bounds(self):
        base = local_source_candidates((-.5, .244), (-.5, .5), (-.264, .264), 9575.,
                                       offsets_m=np.arange(-2000., 2001., 500.))
        extended = extend_boundary_candidates(base, (-.5, .244), (-.5, .5), (-.264, .264), 9575.)
        self.assertTrue(set(base).issubset(extended))
        self.assertEqual(len(extended), 42)
        self.assertTrue(all(-.5 <= x <= .5 and -.264 <= y <= .264 for x,y in extended))
        self.assertTrue(any(y == .264 for _, y in extended))

    def test_absolute_response_scale_is_not_scaled_condition(self):
        response = np.array([[1., 0.], [0., 2.], [1., 1.]])
        q = np.array([3., 4.])
        normal = response_amplitude_diagnostics(response, q)
        weak = response_amplitude_diagnostics(response*.001, q*1000.)
        self.assertAlmostEqual(weak["prediction_l2_normalized"], normal["prediction_l2_normalized"])
        self.assertAlmostEqual(weak["effective_response_gain"], normal["effective_response_gain"]*.001)
        self.assertAlmostEqual(weak["coefficient_l2_model_units"], normal["coefficient_l2_model_units"]*1000.)
        self.assertIsNone(response_amplitude_diagnostics(response, [0., 0.])["effective_response_gain"])
        for bad_response, bad_q in ((response, [-1., 0.]), (response, [1.]), (response*np.nan, q)):
            with self.assertRaises(ValueError):
                response_amplitude_diagnostics(bad_response, bad_q)

    def test_shared_local_candidate_budget_at_boundary(self):
        candidates = local_source_candidates((-.5, .2), (-.5, .5), (-.5, .5), 10000.)
        self.assertEqual(len(candidates), 15)
        self.assertIn((-.5, .2), candidates)
        self.assertTrue(all(-.5 <= x <= .5 and -.5 <= y <= .5 for x, y in candidates))
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertEqual(len(local_source_candidates((-.5, .245), (-.5, .5), (-.264, .264), 9575.)), 9)
        expanded = local_source_candidates((-.5, .245), (-.5, .5), (-.264, .264), 9575.,
            offsets_m=np.arange(-2000., 2001., 500.))
        previous = local_source_candidates((-.5, .245), (-.5, .5), (-.264, .264), 9575.)
        self.assertEqual(len(expanded), 25)
        self.assertTrue(set(previous).issubset(expanded))
        with self.assertRaises(ValueError):
            local_source_candidates((-.6, 0.), (-.5, .5), (-.5, .5), 10000.)
        boundary = .2644534991109576
        rounded = float(np.float32(boundary))
        snapped = local_source_candidates((0., rounded), (-.5, .5), (-boundary, boundary), 10000.)
        self.assertIn((0., boundary), snapped)
        self.assertTrue(all(y <= boundary for _, y in snapped))
        with self.assertRaises(ValueError):
            local_source_candidates((0., boundary+1e-5), (-.5, .5), (-boundary, boundary), 10000.)
    def prototype(self, device="cpu"):
        model = PINN(hidden=8, depth=1).to(device)
        with patch.multiple(field, RECURRENT_ADAPTIVE_SUBSTEPS=False,
                            RECURRENT_SUBSTEPS=1, RECURRENT_MAX_SUBSTEPS=1):
            field.configure_recurrent_context(model, -1., 1., -1., 1.,
                [0., .5, 1., 1.5], [.12, .1, .06, .06], [0., .03, .05, .05],
                d_scale_norm=.002, anisotropic_diffusion=True,
                d_parallel_min_norm=.0005, d_perp_min_norm=.0002,
                decay_norm=.1, nx=21, ny=21)
        return model

    def check_variant(self, device):
        prototype = self.prototype(device)
        knots = prototype.recurrent_times.cpu().numpy()
        observations = torch.tensor([[.1, .1, t] for t in knots], device=device)
        original_flags = (field.RECURRENT_SUBSTEPS, field.RECURRENT_ADAPTIVE_SUBSTEPS)
        original_basis = build_q_basis(prototype.recurrent_integration_times, 4, knots=knots)
        original_response = build_response_matrix(prototype, .1, observations, original_basis)
        for substeps in (1, 2, 4):
            model = configured_variant(prototype, 21, substeps=substeps, preserve_diffusion_mode=True)
            self.assertTrue(model.recurrent_anisotropic_diffusion)
            self.assertEqual(model.recurrent_d_parallel_min_norm, prototype.recurrent_d_parallel_min_norm)
            self.assertEqual(model.recurrent_d_perp_min_norm, prototype.recurrent_d_perp_min_norm)
            self.assertEqual(tuple(model.recurrent_substeps_per_interval), (substeps,)*3)
            self.assertEqual(model.recurrent_integration_times.numel(), 3*substeps+1)
            torch.testing.assert_close(model.recurrent_times, prototype.recurrent_times)
            torch.testing.assert_close(model.recurrent_u, prototype.recurrent_u)
            torch.testing.assert_close(model.recurrent_v, prototype.recurrent_v)
            basis = build_q_basis(model.recurrent_integration_times, 4, knots=knots)
            self.assertEqual(basis.shape, (3*substeps+1, 4))
            torch.testing.assert_close(basis.sum(dim=1), torch.ones(3*substeps+1, device=device))
            np.testing.assert_allclose(basis.cpu().numpy()[::substeps], np.eye(4), atol=1e-7)
            response = build_response_matrix(model, .1, observations, basis)
            self.assertTrue(torch.isfinite(response).all())
            self.assertEqual(response.device, observations.device)
            if substeps == 1:
                torch.testing.assert_close(response, original_response)
        self.assertEqual((field.RECURRENT_SUBSTEPS, field.RECURRENT_ADAPTIVE_SUBSTEPS), original_flags)
        self.assertEqual(tuple(prototype.recurrent_substeps_per_interval), (1,)*3)

    def test_cpu_preserves_physics_and_hourly_controls(self):
        self.check_variant("cpu")

    def test_injection_quadrature_converges_in_calm_gaussian_case(self):
        model = self.prototype()
        model.recurrent_u.zero_()
        model.recurrent_v.zero_()
        model.recurrent_decay_norm = 0.
        knots = model.recurrent_times.cpu().numpy()
        xyt = np.array([[0., 0., 1.5]])
        results = [analytic_release_predictions(model, xyt, (0., 0.), (.03, .03),
                    knots, np.ones(4), .2, n)[0] for n in (1, 2, 4, 256)]
        errors = np.abs(np.asarray(results[:-1])-results[-1])
        self.assertTrue(np.all(np.diff(errors) < 0))
        expected = np.log((.2**2+2*.03*1.5)/.2**2)/(4*np.pi*.03)
        initial_mass = field.RECURRENT_INITIAL_RELEASE_FRACTION*model.recurrent_initial_release_dt
        expected += initial_mass/(2*np.pi*(.2**2+2*.03*1.5))
        self.assertAlmostEqual(results[-1], expected*field.RECURRENT_SOURCE_SCALE, places=4)

    def test_isolated_impulses_preserve_mass_and_measure_diffusion(self):
        model = self.prototype()
        rows = isolated_operator_moments(model, (.002, .002), size=65)
        self.assertEqual(len(rows), 27)
        for row in rows:
            self.assertAlmostEqual(row["retained_mass"], 1., places=6)
            self.assertAlmostEqual(row["center_x_norm"], row["expected_center_x_norm"], places=6)
            self.assertAlmostEqual(row["center_y_norm"], row["expected_center_y_norm"], places=6)
            self.assertGreaterEqual(row["variance_x_norm2"], 0.)
        diffusion = [row for row in rows if row["operator"] == "diffusion_only" and row["interval"] == 0]
        self.assertLess(diffusion[-1]["variance_x_norm2"], diffusion[0]["variance_x_norm2"])
        for row in rows:
            if row["operator"] == "implicit_isotropic_control":
                self.assertAlmostEqual(row["variance_x_norm2"], row["expected_variance_x_norm2"], places=6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_preserves_physics_and_hourly_controls(self):
        self.check_variant("cuda:0")


if __name__ == "__main__":
    unittest.main()
