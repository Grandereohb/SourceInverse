import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pinn_source'))
from tensor_diffusion import build_diffusion_plan, apply_diffusion_plan


class TensorDiffusionTests(unittest.TestCase):
    def grids(self, n=65, device='cpu'):
        x = torch.arange(n, dtype=torch.float64, device=device)*2.
        y = torch.arange(n, dtype=torch.float64, device=device)*.7
        return x, y

    def moments(self, value, x, y):
        yy, xx = torch.meshgrid(y-y[len(y)//2], x-x[len(x)//2], indexing='ij')
        return torch.stack(((value*xx*xx).sum(), (value*xx*yy).sum(), (value*yy*yy).sum()))

    def test_rotated_tensor_covariance_including_underresolved_small_steps(self):
        x, y = self.grids()
        impulse = torch.zeros((65, 65), dtype=x.dtype)
        impulse[32, 32] = 1.
        for u, v in ((1., .8), (1., -.8), (-1., -.8), (0., 0.), (1., 0.)):
            for dt in (1e-4, .03, .5):
                plan = build_diffusion_plan(x, y, .7, .12, u, v, dt)
                result = apply_diffusion_plan(impulse, plan)
                expected = 2*dt*plan.tensor.flatten()[[0, 1, 3]]
                torch.testing.assert_close(self.moments(result, x, y), expected, rtol=2e-7, atol=1e-10)
                self.assertGreaterEqual(result.min(), 0.)
                self.assertAlmostEqual(result.sum().item(), 1., places=8)
                torch.testing.assert_close(result, result.flip((0, 1)), atol=1e-14, rtol=1e-12)

    def test_finite_boundary_against_independent_matrix_exponential(self):
        n = 5
        x, y = self.grids(n)
        plan = build_diffusion_plan(x, y, 3., .5, 1., -1., .3)
        generator = np.zeros((n*n, n*n))
        directions = ((1, 0), (0, 1), plan.direction)
        rates = plan.rates.numpy()
        for iy in range(n):
            for ix in range(n):
                col = iy*n+ix
                generator[col, col] = -2*rates.sum()
                for rate, (sx, sy) in zip(rates, directions):
                    for sign in (-1, 1):
                        jx, jy = ix+sign*sx, iy+sign*sy
                        if 0 <= jx < n and 0 <= jy < n:
                            generator[jy*n+jx, col] += rate
        impulse = torch.zeros((n, n), dtype=x.dtype)
        impulse[0, 0] = 1.
        actual = apply_diffusion_plan(impulse, plan)
        expected = expm(.3*generator) @ impulse.numpy().flatten()
        np.testing.assert_allclose(actual.flatten(), expected, atol=2e-11, rtol=1e-8)
        self.assertLess(actual.sum(), .9)
        self.assertGreater(actual.sum(), 0.)

    def test_semigroup_linearity_and_gradients(self):
        x, y = self.grids(15)
        first = torch.arange(225, dtype=x.dtype).reshape(15, 15)/225
        second = first.flip((0, 1))
        d = torch.tensor(.7, dtype=x.dtype, requires_grad=True)
        plan = build_diffusion_plan(x, y, d, .12, 1., .8, .4)
        half = build_diffusion_plan(x, y, d, .12, 1., .8, .2)
        a = apply_diffusion_plan(first, plan)
        b = apply_diffusion_plan(second, plan)
        torch.testing.assert_close(apply_diffusion_plan(first+second, plan), a+b)
        torch.testing.assert_close(apply_diffusion_plan(apply_diffusion_plan(first, half), half), a, atol=1e-10, rtol=1e-9)
        gradient, = torch.autograd.grad(a.square().sum(), d)
        def objective(dp):
            return apply_diffusion_plan(first, build_diffusion_plan(x, y, dp, .12, 1., .8, .4)).square().sum()
        finite = (objective(.700001)-objective(.699999))/2e-6
        torch.testing.assert_close(gradient, finite, rtol=2e-6, atol=1e-8)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_dtype_device_batch_and_gradient(self):
        x, y = self.grids(15, 'cuda')
        d = torch.tensor(.7, dtype=x.dtype, device=x.device, requires_grad=True)
        plan = build_diffusion_plan(x, y, d, .12, 1., -.8, .3)
        batch = torch.ones((3, 15, 15), dtype=x.dtype, device=x.device)
        result = apply_diffusion_plan(batch, plan)
        self.assertEqual(result.device, batch.device)
        self.assertEqual(result.dtype, batch.dtype)
        self.assertTrue(torch.isfinite(result).all() & (result >= 0).all())
        result.sum().backward()
        self.assertTrue(torch.isfinite(d.grad))
        cx, cy = self.grids(15)
        reference = apply_diffusion_plan(batch.cpu(), build_diffusion_plan(cx, cy, .7, .12, 1., -.8, .3))
        torch.testing.assert_close(result.cpu(), reference, rtol=1e-10, atol=1e-11)

    def test_validation(self):
        x, y = self.grids(5)
        for dp, dq, dt in ((-1., .1, 1.), (1., 0., 1.), (1., .1, -1.), (float('nan'), .1, 1.)):
            with self.assertRaises(ValueError):
                build_diffusion_plan(x, y, dp, dq, 1., 1., dt)

    def test_response_plan_reuse_matches_values_gradients_and_is_fresh(self):
        from unittest.mock import patch
        import field
        import tensor_diffusion
        from test_ia_vpdi import synthetic_case
        from variable_projection import build_response_matrix
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            model, xyt, basis, _, _, _ = synthetic_case(device=device)
            model.recurrent_diffusion_solver = 'tensor_semigroup'
            parameters = (model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta)
            original = tensor_diffusion.build_diffusion_plan
            intervals = model.recurrent_times.numel()-1
            with patch.object(tensor_diffusion, 'build_diffusion_plan', wraps=original) as builder:
                shared = build_response_matrix(model, .08, xyt, basis, track_gradients=True)
                self.assertEqual(builder.call_count, intervals)
                shared_grad = torch.autograd.grad(shared.square().sum(), parameters)
                builder.reset_mock()
                separate = torch.stack([field.recurrent_plume_value_from_q(model, xyt, .08, basis[:, j]).flatten()
                                        for j in range(basis.shape[1])], dim=1)
                self.assertEqual(builder.call_count, intervals*basis.shape[1])
                separate_grad = torch.autograd.grad(separate.square().sum(), parameters)
                torch.testing.assert_close(shared, separate, rtol=0., atol=0.)
                for first, second in zip(shared_grad, separate_grad):
                    torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)
                with torch.no_grad():
                    model.rawD_perp.add_(.3)
                builder.reset_mock()
                updated = build_response_matrix(model, .08, xyt, basis, track_gradients=True)
                self.assertEqual(builder.call_count, intervals)
                self.assertFalse(torch.equal(updated, shared))
                self.assertTrue(all(torch.isfinite(g).all() for g in torch.autograd.grad(updated.square().sum(), parameters)))

    def test_plan_reuse_after_model_device_or_dtype_changes(self):
        import field
        from test_ia_vpdi import synthetic_case
        from variable_projection import build_response_matrix
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            model, xyt, basis, _, _, _ = synthetic_case()
            model.recurrent_diffusion_solver = 'tensor_semigroup'
            model = model.to(device=device, dtype=torch.float64)
            shared = build_response_matrix(model, .08, xyt, basis, track_gradients=True)
            actual = torch.stack([field.recurrent_plume_value_from_q(model, xyt.to(model.xs),
                .08, basis[:, j]).flatten() for j in range(basis.shape[1])], dim=1)
            self.assertEqual(shared.dtype, model.xs.dtype)
            self.assertEqual(shared.device, model.xs.device)
            torch.testing.assert_close(shared, actual, rtol=0., atol=0.)
            shared.sum().backward()
            self.assertTrue(torch.isfinite(model.rawD_perp.grad))

    def test_float32_underresolved_covariance_cpu_and_cuda(self):
        reference = None
        for device in ('cpu', 'cuda:0') if torch.cuda.is_available() else ('cpu',):
            x, y = (axis.float() for axis in self.grids(device=device))
            impulse = torch.zeros((65, 65), dtype=x.dtype, device=x.device)
            impulse[32, 32] = 1.
            plan = build_diffusion_plan(x, y, .007, .0012, 1., -.8, .03)
            result = apply_diffusion_plan(impulse, plan)
            expected = .06*plan.tensor.flatten()[[0, 1, 3]]
            torch.testing.assert_close(self.moments(result, x, y), expected, rtol=2e-5, atol=1e-9)
            self.assertAlmostEqual(result.sum().item(), 1., places=6)
            if reference is None:
                reference = result
            else:
                torch.testing.assert_close(result.cpu(), reference, rtol=2e-5, atol=1e-7)

    def test_stiff_interval_is_positive_and_matches_finite_semigroup(self):
        x, y = self.grids(9)
        field = torch.ones((9, 9), dtype=x.dtype)
        whole = build_diffusion_plan(x, y, 4., .3, 1., -.4, 10.)
        self.assertGreater(whole.repetitions, 1)
        half = build_diffusion_plan(x, y, 4., .3, 1., -.4, 5.)
        once = apply_diffusion_plan(field, whole)
        twice = apply_diffusion_plan(apply_diffusion_plan(field, half), half)
        self.assertTrue(torch.isfinite(once).all() & (once >= 0).all())
        self.assertLess(once.sum(), field.sum())
        torch.testing.assert_close(once, twice, rtol=1e-9, atol=1e-10)
        self.assertLessEqual(whole.tail_bound, 1e-10)

    def test_synthetic_profile_recovers_q_and_improves_source(self):
        from test_ia_vpdi import synthetic_case
        from variable_projection import build_response_matrix
        from ia_vpdi import score_all_data_candidate, refine_profiled_candidate
        model, xyt, basis, coefficients, _, _ = synthetic_case()
        model.recurrent_diffusion_solver = 'tensor_semigroup'
        model.recurrent_source_decay_integrator = 'exponential_linear'
        response = build_response_matrix(model, .08, xyt, basis)
        observations = response @ torch.as_tensor(coefficients, dtype=response.dtype)
        true = score_all_data_candidate(model, .08, xyt, observations, basis, .1, -.1, .004, .002)
        np.testing.assert_allclose(true.coefficients, coefficients, rtol=1e-4, atol=1e-6)
        self.assertLess(true.profile_score, 1e-12)
        start = score_all_data_candidate(model, .08, xyt, observations, basis, -.05, .05, .004, .002)
        refined = refine_profiled_candidate(model, .08, xyt, observations, basis, start,
            (-.8, .8), (-.8, .8), steps=25, lr=.03, patience=15, optimize_diffusion=False)
        self.assertTrue(refined.success)
        self.assertLess(refined.best.profile_score, start.profile_score)
        self.assertLess(np.hypot(refined.best.source_x-.1, refined.best.source_y+.1),
                        np.hypot(start.source_x-.1, start.source_y+.1))

    def test_recurrent_external_q_and_context_rebuild(self):
        import field
        import test_substep_diagnostics as fixtures
        from inversion_validation_common import configured_variant
        for device in ('cpu', 'cuda:0') if torch.cuda.is_available() else ('cpu',):
            model = fixtures.SubstepDiagnosticTests().prototype(device)
            model.recurrent_diffusion_solver = 'tensor_semigroup'
            model.recurrent_source_decay_integrator = 'exponential_linear'
            model = configured_variant(model, 21, preserve_diffusion_mode=True)
            q1 = torch.linspace(.1, 1., len(model.recurrent_integration_times), device=device)
            q2 = q1.flip(0)*.3
            one = field.recurrent_plume_fields_from_q(model, .1, q1)
            two = field.recurrent_plume_fields_from_q(model, .1, q2)
            both = field.recurrent_plume_fields_from_q(model, .1, q1+q2)
            torch.testing.assert_close(both, one+two, atol=3e-7, rtol=3e-6)
            both.sum().backward()
            self.assertTrue(torch.isfinite(model.rawD_parallel_delta.grad))
            sampled = field.recurrent_plume_fields_at_times(model, .1, model.recurrent_times,
                                                            model.recurrent_u, model.recurrent_v)
            self.assertEqual(model.recurrent_diffusion_solver, 'tensor_semigroup')
            self.assertTrue(torch.isfinite(sampled).all() & (sampled >= 0).all())


if __name__ == '__main__':
    unittest.main()
