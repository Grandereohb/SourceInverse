import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pinn_source'))
import field
from models.pinn import PINN
from variable_projection import build_q_basis, build_response_matrix, solve_nonnegative_q, inversion_timing
from ia_vpdi import _prediction_jacobian_raw


class BatchedResponseTests(unittest.TestCase):
    def model(self, device, solver, anisotropic, integrator, steps=2):
        model = PINN(hidden=8, depth=1).to(device=device, dtype=torch.float64)
        with patch.multiple(field, RECURRENT_ADAPTIVE_SUBSTEPS=False,
                            RECURRENT_SUBSTEPS=steps, RECURRENT_MAX_SUBSTEPS=steps):
            field.configure_recurrent_context(model, -1., 1., -1., 1., [0., .3, .7, 1.],
                [.2, 0., -.1, .1], [.1, 0., .2, .2], d_scale_norm=.002,
                anisotropic_diffusion=anisotropic, source_mode='cell_gaussian',
                source_sigma_norm=.12, diffusion_solver=solver,
                source_decay_integrator=integrator, decay_norm=.2, nx=12, ny=10)
        return model

    def compare(self, device):
        for solver, anisotropic in (('gaussian', True), ('gaussian', False),
                                    ('implicit', False), ('tensor_semigroup', True)):
            for integrator in ('legacy', 'exponential_linear'):
                with self.subTest(device=device, solver=solver, integrator=integrator):
                    model = self.model(device, solver, anisotropic, integrator)
                    basis = build_q_basis(model.recurrent_integration_times.to(model.xs), 5)
                    query = model.xs.new_tensor([[.1, .1, 0.], [0., .2, .3],
                        [-.1, .1, .47], [.4, -.3, 1.], [2., 0., .8]])
                    parameters = [model.xs, model.ys]
                    parameters += ([model.rawD_perp, model.rawD_parallel_delta]
                                   if anisotropic else [model.rawD])
                    reference = build_response_matrix(model, .1, query, basis,
                        track_gradients=True, batch_size=1)
                    ref_grad = torch.autograd.grad(reference.square().sum(), parameters)
                    for size in (2, 8):
                        actual = build_response_matrix(model, .1, query, basis,
                            track_gradients=True, batch_size=size)
                        actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
                        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-11)
                        for left, right in zip(actual_grad, ref_grad):
                            torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-10)
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertTrue((actual >= 0).all())
                    target = reference.detach() @ model.xs.new_tensor([1., .5, 2., 1., .3])
                    first = solve_nonnegative_q(reference, target)
                    second = solve_nonnegative_q(actual, target)
                    self.assertAlmostEqual(first.weighted_sse, second.weighted_sse, places=10)

    def test_cpu_value_gradient_and_nnls_equivalence(self):
        self.compare('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_value_gradient_and_nnls_equivalence(self):
        self.compare('cuda')

    def test_kernels_shared_and_refreshed_after_parameter_update(self):
        model = self.model('cpu', 'gaussian', True, 'legacy', steps=3)
        basis = build_q_basis(model.recurrent_integration_times, 5)
        query = model.xs.new_tensor([[.1, .1, .6], [0., 0., 1.]])
        original = field._build_anisotropic_gaussian_plan
        with patch.object(field, '_build_anisotropic_gaussian_plan', wraps=original) as builder:
            actual = build_response_matrix(model, .1, query, basis, batch_size=2)
            self.assertEqual(builder.call_count, 3)
            builder.reset_mock()
            with torch.no_grad():
                model.rawD_perp.add_(1.)
            updated = build_response_matrix(model, .1, query, basis, batch_size=2)
            self.assertEqual(builder.call_count, 3)
            self.assertFalse(torch.equal(actual, updated))
        for value in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                build_response_matrix(model, .1, query, basis, batch_size=value)

    def test_batched_diagnostic_jacobian_matches_serial_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            model = self.model(device, 'gaussian', True, 'legacy')
            basis = build_q_basis(model.recurrent_integration_times.to(model.xs), 5)
            query = model.xs.new_tensor([[0., 0., .2], [.2, .1, .5], [.1, .2, 1.]])
            response = build_response_matrix(model, .1, query, basis, track_gradients=True)
            prediction = response @ model.xs.new_tensor([1., 2., .5, .1, 1.])
            serial = _prediction_jacobian_raw(model, prediction, batch_size=1)
            for size in (2, 16):
                actual = _prediction_jacobian_raw(model, prediction, batch_size=size)
                torch.testing.assert_close(actual, serial, rtol=1e-8, atol=1e-10)

    def test_near_clock_queries_preserve_exact_time_sampling(self):
        model = self.model('cpu', 'gaussian', True, 'legacy')
        basis = build_q_basis(model.recurrent_integration_times.to(model.xs), 5)
        query = model.xs.new_tensor([[0., 0., .3-2e-7], [.2, .1, .7-2e-7]])
        reference = build_response_matrix(model, .1, query, basis, batch_size=1)
        actual = build_response_matrix(model, .1, query, basis, batch_size=8)
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-11)

    def test_component_timing_is_scoped_and_can_be_disabled(self):
        model = self.model('cpu', 'gaussian', True, 'legacy')
        basis = build_q_basis(model.recurrent_integration_times.to(model.xs), 5)
        query = model.xs.new_tensor([[0., 0., .5]])
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'PINN_IA_VP_TIMING': enabled}), inversion_timing() as record:
                response = build_response_matrix(model, .1, query, basis)
                solve_nonnegative_q(response, [1.])
            if enabled == '0':
                self.assertFalse(record['stages'])
            else:
                self.assertEqual(record['stages']['response_matrix']['calls'], 1)
                self.assertEqual(record['stages']['inner_q_solve']['calls'], 1)


if __name__ == '__main__':
    unittest.main()
