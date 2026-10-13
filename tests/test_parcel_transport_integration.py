from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pinn_source'))
import field
from models.pinn import PINN
from variable_projection import build_q_basis, build_response_matrix, solve_nonnegative_q


class ParcelTransportIntegrationTests(unittest.TestCase):
    def model(self, n=12, mode='gaussian_parcels', device='cpu'):
        model = PINN(hidden=8, depth=1).to(device)
        field.configure_recurrent_context(model, -1., 1., -1., 1., np.linspace(0., 1., 6),
            [.2]*6, [.05]*6, source_mode='cell_gaussian', source_sigma_norm=.08,
            anisotropic_diffusion=True, d_parallel_min_norm=.001, d_perp_min_norm=.001,
            d_scale_norm=.002, decay_norm=.1, transport_solver=mode,
            parcel_subdivisions=16, nx=n, ny=n)
        return model

    def check_direct(self, device):
        model = self.model(device=device)
        clock = model.recurrent_integration_times.to(device)
        query = torch.tensor([[.1, .02, .13], [.2, .03, .7], [0., 0., 1.]], device=device)
        q1, q2 = .2+clock, .7-clock*.2
        a = field.recurrent_plume_value_from_q(model, query, .05, q1)
        b = field.recurrent_plume_value_from_q(model, query, .05, q2)
        c = field.recurrent_plume_value_from_q(model, query, .05, q1+q2)
        torch.testing.assert_close(c, a+b, rtol=2e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(c).all())
        self.assertTrue((c >= 0).all())
        self.assertEqual(c.device.type, device)
        model.configure_external_q(clock, q1)
        torch.testing.assert_close(field.recurrent_plume_value(model, query, .05), a)
        basis = build_q_basis(clock, 3, knots=[0., .5, 1.])
        response = build_response_matrix(model, .05, query, basis, track_gradients=True)
        torch.testing.assert_close(response @ torch.tensor([1., 2., 1.], device=device),
            field.recurrent_plume_value_from_q(model, query, .05, basis @ torch.tensor([1., 2., 1.], device=device)).reshape(-1))
        response.sum().backward()
        self.assertTrue(torch.isfinite(model.xs.grad))
        self.assertTrue(torch.isfinite(model.rawD_perp.grad))

    def test_linear_direct_response_and_gradients_cpu(self):
        self.check_direct('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_linear_direct_response_and_gradients_cuda(self):
        self.check_direct('cuda')

    def test_q_clock_and_point_predictions_independent_of_raster(self):
        small, large = self.model(12), self.model(24)
        torch.testing.assert_close(small.recurrent_integration_times, large.recurrent_integration_times)
        self.assertIn(.5, small.recurrent_integration_times.tolist())
        query = torch.tensor([[.05, .05, .35], [.1, .02, .9]])
        q = .5+small.recurrent_integration_times
        torch.testing.assert_close(field.recurrent_plume_value_from_q(small, query, .05, q),
            field.recurrent_plume_value_from_q(large, query, .05, q))
        self.assertEqual(small.recurrent_substeps_per_interval, large.recurrent_substeps_per_interval)

    def test_render_coordinates_use_original_wind_clock_and_restore_state(self):
        model = self.model()
        clock = model.recurrent_integration_times
        model.configure_external_q(clock, .5+clock)
        xs, ys, times = [-.1, .05, .2], [-.03, .02], [.13, .7]
        saved = model.recurrent_times
        images = field.recurrent_plume_fields_at_times(model, .05, times, [999., 999.], [999., 999.],
            x_values=xs, y_values=ys)
        self.assertEqual(tuple(images.shape), (2, 2, 3))
        yy, xx = torch.meshgrid(torch.tensor(ys), torch.tensor(xs), indexing='ij')
        query = torch.stack((xx.flatten(), yy.flatten(), torch.full((6,), times[0])), dim=1)
        torch.testing.assert_close(images[0].flatten(), field.recurrent_plume_value(model, query, .05).flatten())
        self.assertIs(model.recurrent_times, saved)
        with patch.object(field, '_recurrent_plume_fields_core', side_effect=AssertionError('grid branch used')):
            self.assertTrue(torch.isfinite(field.recurrent_plume_fields(model, .05)).all())

    def test_recovery_and_invalid_mixed_contexts(self):
        model = self.model()
        clock = model.recurrent_integration_times
        basis = build_q_basis(clock, 3, knots=[0., .5, 1.])
        query = torch.stack((torch.full((6,), .05), torch.full((6,), .02), model.recurrent_times), dim=1)
        matrix = build_response_matrix(model, .05, query, basis)
        truth = np.array([.7, 1.2, .4])
        fit = solve_nonnegative_q(matrix, matrix.numpy() @ truth)
        np.testing.assert_allclose(fit.coefficients, truth, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, 'separate grid context'):
            build_response_matrix(model, .05, query, basis, transport_solver='grid')
        with self.assertRaises(ValueError):
            field.recurrent_plume_fields_from_q(model, .05, torch.ones(len(clock)), initial_field=torch.zeros(12, 12))
        with self.assertRaises(ValueError):
            field.recurrent_plume_value_from_q(model, query, .05, -torch.ones(len(clock)))


if __name__ == '__main__':
    unittest.main()
