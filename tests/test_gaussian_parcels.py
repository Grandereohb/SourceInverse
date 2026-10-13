import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))
sys.path.insert(0, str(ROOT / "scripts"))
from models.pinn import PINN
from field import configure_recurrent_context
from gaussian_parcels import gaussian_parcel_response, gaussian_parcel_value_from_q
from variable_projection import build_q_basis, build_response_matrix
from inversion_validation_common import analytic_release_predictions


class GaussianParcelTests(unittest.TestCase):
    def setup_case(self, device="cpu"):
        model = PINN(hidden=8, depth=1).to(device=device, dtype=torch.float64)
        configure_recurrent_context(model, -1., 1., -1., 1.,
            [0., .5, 1.], [.3, 0., 0.], [0., .3, .3],
            d_scale_norm=.01, anisotropic_diffusion=True,
            d_parallel_min_norm=.01, d_perp_min_norm=.005,
            decay_norm=.1, source_sigma_norm=.15, nx=15, ny=15)
        with torch.no_grad():
            model.xs.zero_()
            model.ys.zero_()
        xyt = torch.tensor([[.1, .05, 0.], [.15, .1, .5], [.15, .15, 1.]],
                           dtype=model.xs.dtype, device=device)
        return model, xyt

    def test_linearity_nonnegativity_and_no_grid_dependence(self):
        model, xyt = self.setup_case()
        times = model.recurrent_times
        a = times.new_tensor([1., 2., .1])
        b = times.new_tensor([.2, .1, 3.])
        predict = lambda q: gaussian_parcel_value_from_q(model, xyt, .15, times, q)
        torch.testing.assert_close(predict(a+b), predict(a)+predict(b))
        self.assertTrue(bool((predict(a) >= 0).all()))
        before = predict(a)
        model.recurrent_x_grid *= 2
        torch.testing.assert_close(predict(a), before)

    def test_rotating_anisotropic_wind_matches_independent_reference(self):
        model, xyt = self.setup_case()
        a, b = model.anisotropic_D()
        along = float(model.recurrent_d_parallel_min_norm+a.detach()*model.recurrent_d_scale_norm)
        across = float(model.recurrent_d_perp_min_norm+b.detach()*model.recurrent_d_scale_norm)
        along = max(along, across)
        q = np.array([1., 2., .1])
        expected = analytic_release_predictions(model, xyt.numpy(), (0., 0.), (along, across),
                                               model.recurrent_times.numpy(), q, .15, 32)
        actual = gaussian_parcel_value_from_q(model, xyt, .15, model.recurrent_times, q, subdivisions=32)
        np.testing.assert_allclose(actual.detach().numpy().ravel(), expected, rtol=1e-6, atol=1e-7)

    def test_response_interface_gradients_and_grid_default(self):
        model, xyt = self.setup_case()
        basis = build_q_basis(model.recurrent_integration_times, 3)
        actual = build_response_matrix(model, .15, xyt, basis, True, transport_solver="gaussian_parcels")
        actual.sum().backward()
        for parameter in (model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        torch.testing.assert_close(build_response_matrix(model, .15, xyt, basis),
                                   build_response_matrix(model, .15, xyt, basis, transport_solver="grid"))

    def test_validation(self):
        model, xyt = self.setup_case()
        basis = torch.eye(3, dtype=model.xs.dtype)
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                gaussian_parcel_response(model, .15, xyt, model.recurrent_times, basis, steps)
        with self.assertRaises(ValueError):
            gaussian_parcel_response(model, .15, xyt+torch.tensor([0., 0., 2.]), model.recurrent_times, basis)
        model.recurrent_source_mode = "point"
        with self.assertRaises(ValueError):
            gaussian_parcel_response(model, .15, xyt, model.recurrent_times, basis)

    def test_release_quadrature_converges_without_refitting_q(self):
        model, xyt = self.setup_case()
        basis = torch.eye(3, dtype=model.xs.dtype)
        values = [gaussian_parcel_response(model, .15, xyt, model.recurrent_times, basis,
                    subdivisions=n).detach() for n in (4, 16, 64, 256)]
        errors = [float(torch.linalg.vector_norm(value-values[-1])) for value in values[:-1]]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_agreement(self):
        cpu, xyt = self.setup_case()
        gpu, gpu_xyt = self.setup_case("cuda:0")
        gpu.load_state_dict(cpu.state_dict())
        basis = torch.eye(3, dtype=cpu.xs.dtype)
        expected = gaussian_parcel_response(cpu, .15, xyt, cpu.recurrent_times, basis)
        actual = gaussian_parcel_response(gpu, .15, gpu_xyt, gpu.recurrent_times, basis)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
