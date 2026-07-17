import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from field import _advect_field, _diffuse_field  # noqa: E402
from transport_units import (  # noqa: E402
    normalize_decay_per_hour,
    normalize_diffusivity_m2s,
    normalize_velocity_mps,
)


class TransportUnitTests(unittest.TestCase):
    def test_physical_unit_normalization(self):
        velocity = normalize_velocity_mps(1.0, 2.0, 7200.0, factor=0.25)
        diffusion = normalize_diffusivity_m2s(2.0, 2.0, 7200.0)
        decay = normalize_decay_per_hour(0.05, 12.0)

        self.assertAlmostEqual(float(velocity), 0.25, places=7)
        self.assertAlmostEqual(float(diffusion), 1.0 / 3600.0, places=10)
        self.assertAlmostEqual(decay, 0.6, places=7)

    def test_advection_moves_field_in_wind_direction(self):
        grid = torch.linspace(-1.0, 1.0, 41)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        field = torch.zeros((41, 41))
        field[20, 20] = 1.0

        moved = _advect_field(
            field,
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            torch.tensor(0.1),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )
        mass = moved.sum()
        x_center = (moved * xx).sum() / mass
        y_center = (moved * yy).sum() / mass

        self.assertAlmostEqual(float(x_center), 0.1, places=5)
        self.assertAlmostEqual(float(y_center), 0.0, places=5)

    def test_advection_uses_open_boundaries(self):
        grid = torch.linspace(-1.0, 1.0, 41)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        field = torch.zeros((41, 41))
        field[20, 39] = 1.0

        moved = _advect_field(
            field,
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            torch.tensor(0.2),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        self.assertLess(float(moved.sum()), 1e-6)

    def test_gaussian_diffusion_matches_expected_variance(self):
        grid = torch.linspace(-0.5, 0.5, 41)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        field = torch.zeros((41, 41))
        field[20, 20] = 1.0
        diffusion = torch.tensor(0.001)
        dt = torch.tensor(1.0)

        diffused = _diffuse_field(field, grid, grid, diffusion, dt)
        mass = diffused.sum()
        radial_variance = (diffused * (xx**2 + yy**2)).sum() / mass

        self.assertAlmostEqual(float(mass), 1.0, places=5)
        self.assertAlmostEqual(float(radial_variance), 0.004, delta=0.0005)


if __name__ == "__main__":
    unittest.main()
