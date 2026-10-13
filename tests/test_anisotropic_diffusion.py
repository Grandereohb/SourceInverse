import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from config import ENABLE_ANISOTROPIC_DIFFUSION  # noqa: E402
from field import (  # noqa: E402
    _diffuse_field,
    _diffuse_field_anisotropic,
    configure_recurrent_context,
    recurrent_plume_fields_from_q,
)
from models.pinn import PINN  # noqa: E402
from pipeline import _freeze_recurrent_unused_parameters  # noqa: E402


def _point_field(size=81):
    grid = torch.linspace(-1.0, 1.0, size)
    field = torch.zeros((size, size))
    field[size // 2, size // 2] = 1.0
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    return grid, field, xx, yy


def _variance(field, coordinate):
    mass = field.sum()
    center = (field * coordinate).sum() / mass
    return (field * (coordinate - center) ** 2).sum() / mass


class AnisotropicDiffusionTests(unittest.TestCase):
    def test_parameterization_enforces_parallel_not_less_than_perpendicular(self):
        model = PINN(hidden=8, depth=1)
        for raw_perp, raw_delta in [(-8.0, -8.0), (0.0, -2.0), (3.0, 4.0)]:
            with torch.no_grad():
                model.rawD_perp.fill_(raw_perp)
                model.rawD_parallel_delta.fill_(raw_delta)
            d_parallel, d_perp = model.anisotropic_D()
            self.assertGreater(float(d_perp.detach()), 0.0)
            self.assertGreaterEqual(
                float(d_parallel.detach()), float(d_perp.detach())
            )

    def test_disabled_mode_freezes_only_anisotropic_parameters(self):
        self.assertFalse(ENABLE_ANISOTROPIC_DIFFUSION)
        model = PINN(hidden=8, depth=1)
        frozen = _freeze_recurrent_unused_parameters(model, "joint_adam")
        self.assertIn("rawD_perp", frozen)
        self.assertIn("rawD_parallel_delta", frozen)
        self.assertTrue(model.rawD.requires_grad)
        self.assertFalse(model.rawD_perp.requires_grad)
        self.assertFalse(model.rawD_parallel_delta.requires_grad)

    def test_enabled_mode_freezes_scalar_diffusion_parameter(self):
        model = PINN(hidden=8, depth=1)
        with patch("pipeline.ENABLE_ANISOTROPIC_DIFFUSION", True):
            frozen = _freeze_recurrent_unused_parameters(model, "joint_adam")
        self.assertIn("rawD", frozen)
        self.assertFalse(model.rawD.requires_grad)
        self.assertTrue(model.rawD_perp.requires_grad)
        self.assertTrue(model.rawD_parallel_delta.requires_grad)

    def test_mass_and_axis_symmetry_are_preserved_away_from_boundaries(self):
        grid, field, _, _ = _point_field()
        diffused = _diffuse_field_anisotropic(
            field,
            grid,
            grid,
            diffusion_parallel=torch.tensor(0.01),
            diffusion_perp=torch.tensor(0.002),
            u=torch.tensor(1.0),
            v=torch.tensor(0.0),
            dt=torch.tensor(1.0),
        )
        self.assertAlmostEqual(float(diffused.sum()), 1.0, places=5)
        self.assertTrue(torch.allclose(diffused, torch.flip(diffused, dims=[0])))
        self.assertTrue(torch.allclose(diffused, torch.flip(diffused, dims=[1])))

    def test_downwind_variance_exceeds_crosswind_variance(self):
        grid, field, xx, yy = _point_field()
        diffused = _diffuse_field_anisotropic(
            field,
            grid,
            grid,
            diffusion_parallel=torch.tensor(0.012),
            diffusion_perp=torch.tensor(0.002),
            u=torch.tensor(1.0),
            v=torch.tensor(0.0),
            dt=torch.tensor(1.0),
        )
        variance_parallel = _variance(diffused, xx)
        variance_perp = _variance(diffused, yy)
        self.assertGreater(float(variance_parallel), 4.0 * float(variance_perp))

    def test_rotated_kernel_follows_diagonal_wind(self):
        grid, field, xx, yy = _point_field()
        diffused = _diffuse_field_anisotropic(
            field,
            grid,
            grid,
            diffusion_parallel=torch.tensor(0.012),
            diffusion_perp=torch.tensor(0.002),
            u=torch.tensor(1.0),
            v=torch.tensor(1.0),
            dt=torch.tensor(1.0),
        )
        coordinate_parallel = (xx + yy) / torch.sqrt(torch.tensor(2.0))
        coordinate_perp = (-xx + yy) / torch.sqrt(torch.tensor(2.0))
        self.assertGreater(
            float(_variance(diffused, coordinate_parallel)),
            4.0 * float(_variance(diffused, coordinate_perp)),
        )

    def test_equal_components_match_isotropic_gaussian(self):
        grid, field, _, _ = _point_field(size=51)
        isotropic = _diffuse_field(
            field, grid, grid, torch.tensor(0.004), torch.tensor(1.0)
        )
        anisotropic = _diffuse_field_anisotropic(
            field,
            grid,
            grid,
            diffusion_parallel=torch.tensor(0.004),
            diffusion_perp=torch.tensor(0.004),
            u=torch.tensor(0.6),
            v=torch.tensor(0.8),
            dt=torch.tensor(1.0),
        )
        self.assertTrue(torch.allclose(isotropic, anisotropic, rtol=2e-5, atol=2e-7))

    def test_anisotropic_recurrent_path_is_finite(self):
        model = PINN(hidden=8, depth=1)
        configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=[0.0, 1.0],
            u_values=[0.1, 0.1],
            v_values=[0.05, 0.05],
            d_min_norm=0.0,
            d_scale_norm=0.002,
            anisotropic_diffusion=True,
            d_parallel_min_norm=0.001,
            d_perp_min_norm=0.0005,
            decay_norm=0.1,
            nx=21,
            ny=21,
        )
        q_values = torch.ones(model.recurrent_integration_times.numel())
        fields = recurrent_plume_fields_from_q(model, 0.08, q_values)
        self.assertTrue(torch.isfinite(fields).all())
        self.assertGreater(float(fields.detach().sum()), 0.0)
        weighted_field = fields[-1] * torch.linspace(
            0.5, 1.5, fields[-1].numel()
        ).view_as(fields[-1])
        weighted_field.sum().backward()
        self.assertIsNotNone(model.rawD_perp.grad)
        self.assertIsNotNone(model.rawD_parallel_delta.grad)
        self.assertTrue(torch.isfinite(model.rawD_perp.grad))
        self.assertTrue(torch.isfinite(model.rawD_parallel_delta.grad))

    def test_ia_vpdi_freezes_scalar_and_keeps_anisotropic_parameters(self):
        model = PINN(hidden=8, depth=1)
        frozen = _freeze_recurrent_unused_parameters(model, "ia_vpdi")
        self.assertIn("rawD", frozen)
        self.assertFalse(model.rawD.requires_grad)
        self.assertTrue(model.rawD_perp.requires_grad)
        self.assertTrue(model.rawD_parallel_delta.requires_grad)


if __name__ == "__main__":
    unittest.main()
