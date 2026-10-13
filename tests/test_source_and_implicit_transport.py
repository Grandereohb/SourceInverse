import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pinn_source"))
import field
from models.pinn import PINN


def model_context(mode="gaussian", solver="gaussian", width=None, device="cpu"):
    model = PINN(hidden=8, depth=1).to(device)
    field.configure_recurrent_context(model, -1., 1., -1., 1.,
        [0., .5, 1.], [.1, .1, .1], [0., 0., 0.],
        d_min_norm=0., d_scale_norm=.002, anisotropic_diffusion=False,
        nx=21, ny=21, source_mode=mode, source_sigma_norm=width,
        diffusion_solver=solver)
    return model


def source(model, sigma=.08):
    return field._source_grid(model, model.recurrent_times[0],
        model.recurrent_x_grid, model.recurrent_y_grid,
        model.recurrent_x_mesh, model.recurrent_y_mesh, sigma)


class SourceAndImplicitTests(unittest.TestCase):
    def test_legacy_source_exact_equivalence(self):
        model = model_context()
        xs, ys = model.source_xy(torch.zeros((1, 1)))
        grid = torch.exp(-((model.recurrent_x_mesh-xs)**2 +
                           (model.recurrent_y_mesh-ys)**2)/(2*.08**2))
        dx = model.recurrent_x_grid[1]-model.recurrent_x_grid[0]
        expected = grid / torch.clamp(grid.sum()*dx*dx, min=1e-8)
        torch.testing.assert_close(source(model), expected, rtol=0, atol=0)

    def test_point_mass_centroid_and_position_gradient(self):
        model = model_context("point")
        with torch.no_grad():
            model.xs.fill_(.035)
            model.ys.fill_(-.047)
        values = source(model)
        dx = model.recurrent_x_grid[1]-model.recurrent_x_grid[0]
        torch.testing.assert_close(values.sum()*dx*dx, torch.tensor(1.))
        centroid = (values*model.recurrent_x_mesh).sum()*dx*dx
        torch.testing.assert_close(centroid, model.xs.reshape(()))
        centroid.backward()
        torch.testing.assert_close(model.xs.grad, torch.ones_like(model.xs))
        with torch.no_grad():
            model.xs.fill_(1.01)
        with self.assertRaisesRegex(ValueError, "outside"):
            source(model)

    def test_cell_gaussian_boundary_mass_not_renormalized(self):
        model = model_context("cell_gaussian", width=.2)
        with torch.no_grad():
            model.xs.fill_(1.)
            model.ys.fill_(0.)
        values = source(model)
        dx = model.recurrent_x_grid[1]-model.recurrent_x_grid[0]
        total = values.sum()*dx*dx
        expected = .5*(torch.erf(torch.tensor(.05/(2**.5*.2))) + 1.)
        self.assertAlmostEqual(float(total.detach()), float(expected), places=5)
        self.assertLess(float(total.detach()), .7)
        total.backward()
        self.assertTrue(torch.isfinite(model.xs.grad).all())

    def test_explicit_narrow_source_keeps_physical_width(self):
        model = model_context("cell_gaussian", width=1e-5)
        with torch.no_grad():
            model.xs.fill_(0.)
            model.ys.fill_(0.)
        values = source(model)
        dx = model.recurrent_x_grid[1]-model.recurrent_x_grid[0]
        torch.testing.assert_close(values.sum()*dx*dx, torch.tensor(1.))
        self.assertEqual(int((values > 0).sum()), 1)

    def test_small_step_variance_and_diffusion_gradient(self):
        grid = torch.linspace(-2., 2., 81, dtype=torch.float64)
        initial = torch.zeros((81, 81), dtype=grid.dtype)
        initial[40, 40] = 1.
        diffusion = torch.tensor(.0002, dtype=grid.dtype, requires_grad=True)
        result = field._diffuse_field_implicit(initial, grid, grid, diffusion, torch.tensor(.1))
        variance = (result*grid[None, :]**2).sum()
        self.assertAlmostEqual(float(variance.detach()), 2*.0002*.1, places=10)
        self.assertAlmostEqual(float(result.sum().detach()), 1., places=10)
        variance.backward()
        self.assertAlmostEqual(float(diffusion.grad), .2, places=7)
        sampled = field._diffuse_field(initial, grid, grid, diffusion.detach(), torch.tensor(.1))
        self.assertLess(float((sampled*grid[None, :]**2).sum()), float(variance.detach())/100)

    def test_substeps_preserve_interior_diffusion_variance(self):
        grid = torch.linspace(-2., 2., 81, dtype=torch.float64)
        result = torch.zeros((81, 81), dtype=grid.dtype)
        result[40, 40] = 1.
        for _ in range(4):
            result = field._diffuse_field_implicit(result, grid, grid,
                torch.tensor(.0002, dtype=grid.dtype), torch.tensor(.025, dtype=grid.dtype))
        self.assertAlmostEqual(float((result*grid[None, :]**2).sum()), 2*.0002*.1, places=10)

    def test_dense_output_preserves_and_restores_mechanism_context(self):
        model = model_context("cell_gaussian", "implicit", .1)
        saved = model.recurrent_times
        result = field.recurrent_plume_fields_at_times(model, .08,
            [0., .25, .5, .75, 1.], [.1]*5, [0.]*5)
        self.assertEqual(result.shape[0], 5)
        self.assertTrue(torch.isfinite(result).all())
        self.assertIs(model.recurrent_times, saved)
        self.assertEqual(model.recurrent_source_mode, "cell_gaussian")
        self.assertEqual(model.recurrent_source_sigma_norm, .1)
        self.assertEqual(model.recurrent_diffusion_solver, "implicit")

    def test_large_step_positive_absorbing_and_finite(self):
        grid = torch.linspace(-1., 1., 21, dtype=torch.float64)
        initial = torch.zeros((21, 21), dtype=grid.dtype)
        initial[0, 0] = 1.
        result = field._diffuse_field_implicit(initial, grid, grid,
            torch.tensor(10., dtype=grid.dtype), torch.tensor(1., dtype=grid.dtype))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.)
        self.assertLess(float(result.sum()), 1.)

    def check_linearity(self, device):
        for mode in ("point", "cell_gaussian"):
            model = model_context(mode, "implicit", .1 if mode == "cell_gaussian" else None, device)
            q1 = torch.ones_like(model.recurrent_integration_times)
            q2 = torch.linspace(.1, .8, len(q1), device=device)
            f = lambda q: field.recurrent_plume_fields_from_q(model, .08, q)
            torch.testing.assert_close(f(q1+q2), f(q1)+f(q2), rtol=3e-5, atol=1e-5)
            self.assertTrue(torch.isfinite(f(q1)).all())

    def test_external_q_linearity_cpu(self):
        self.check_linearity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_external_q_linearity_cuda(self):
        self.check_linearity("cuda")

    def test_implicit_rejects_rotated_tensor(self):
        model = PINN(hidden=8, depth=1)
        with self.assertRaisesRegex(ValueError, "scalar"):
            field.configure_recurrent_context(model, -1., 1., -1., 1.,
                [0., 1.], [0., 0.], [0., 0.], anisotropic_diffusion=True,
                diffusion_solver="implicit")


if __name__ == "__main__":
    unittest.main()
