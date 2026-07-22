import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from field import (  # noqa: E402
    _advance_recurrent_step,
    _advect_field,
    _build_adaptive_substep_plan,
    _build_bilinear_sample_plan,
    _diffuse_field,
    configure_recurrent_context,
    recurrent_plume_fields,
    recurrent_plume_fields_at_times,
)
from transport_units import (  # noqa: E402
    normalize_decay_per_hour,
    normalize_diffusivity_m2s,
    normalize_velocity_mps,
)


class TransportUnitTests(unittest.TestCase):
    class _ConstantSourceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.xs = torch.nn.Parameter(torch.tensor(0.0))
            self.ys = torch.nn.Parameter(torch.tensor(0.0))
            self.q_call_count = 0
            self.source_xy_call_count = 0

        def D(self):
            return torch.tensor(0.0, dtype=self.xs.dtype, device=self.xs.device)

        def Q(self, t):
            self.q_call_count += 1
            return torch.ones_like(t)

        def source_xy(self, t):
            self.source_xy_call_count += 1
            return self.xs.expand_as(t), self.ys.expand_as(t)

    def test_physical_unit_normalization(self):
        velocity = normalize_velocity_mps(1.0, 2.0, 7200.0, factor=0.25)
        diffusion = normalize_diffusivity_m2s(2.0, 2.0, 7200.0)
        decay = normalize_decay_per_hour(0.05, 12.0)

        self.assertAlmostEqual(float(velocity), 0.25, places=7)
        self.assertAlmostEqual(float(diffusion), 1.0 / 3600.0, places=10)
        self.assertAlmostEqual(decay, 0.6, places=7)

    def test_adaptive_substeps_limit_advection_to_one_cell(self):
        selected, required, distances = _build_adaptive_substep_plan(
            t_values=[0.0, 1.0, 2.0],
            u_values=[0.1, 0.025, 0.0],
            v_values=[0.0, 0.0, 0.0],
            dx=0.05,
            dy=0.05,
            minimum_substeps=1,
            max_advection_cells=1.0,
            maximum_substeps=16,
        )

        self.assertEqual(selected, (2, 1))
        self.assertEqual(required, (2, 1))
        for distance, substeps in zip(distances, selected):
            self.assertLessEqual(distance / substeps, 1.0 + 1e-9)

    def test_adaptive_substeps_respect_safety_cap(self):
        selected, required, _ = _build_adaptive_substep_plan(
            t_values=[0.0, 1.0],
            u_values=[1.0, 1.0],
            v_values=[0.0, 0.0],
            dx=0.05,
            dy=0.05,
            minimum_substeps=1,
            max_advection_cells=1.0,
            maximum_substeps=8,
        )

        self.assertEqual(selected, (8,))
        self.assertEqual(required, (20,))

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

    def test_precomputed_advection_matches_direct_sampling(self):
        grid = torch.linspace(-1.0, 1.0, 41)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        field = torch.exp(-10.0 * (xx**2 + yy**2))
        dt = torch.tensor(0.4)
        u = torch.tensor(0.13)
        v = torch.tensor(-0.07)
        plan = _build_bilinear_sample_plan(
            xx.reshape(-1) - u * dt,
            yy.reshape(-1) - v * dt,
            grid,
            grid,
        )

        direct = _advect_field(
            field, grid, grid, xx.reshape(-1), yy.reshape(-1), u, v, dt
        )
        cached = _advect_field(
            field,
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            u,
            v,
            dt,
            sample_plan=plan,
        )

        self.assertTrue(torch.equal(direct, cached))

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

    def test_symmetric_source_injection_keeps_fresh_mass_at_source(self):
        grid = torch.linspace(-1.0, 1.0, 41)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        field = torch.zeros((41, 41))
        source = torch.zeros((41, 41))
        source[20, 20] = 1.0

        advanced = _advance_recurrent_step(
            field,
            source,
            torch.tensor(1.0),
            source,
            torch.tensor(1.0),
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            torch.tensor(0.2),
            torch.tensor(0.0),
            torch.tensor(1e-12),
            0.0,
            1.0,
            torch.tensor(1.0),
        )

        self.assertGreater(float(advanced[20, 20]), 0.49)
        self.assertGreater(float(advanced[20, 24]), 0.49)

    def test_initial_field_is_stored_before_advection(self):
        model = self._ConstantSourceModel()
        configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=[0.0, 1.0],
            u_values=[0.2, 0.2],
            v_values=[0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            nx=41,
            ny=41,
        )

        fields = recurrent_plume_fields(model, sigma_src=0.02)
        first = fields[0]
        grid = model.recurrent_x_grid
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        first_x_center = (first * xx).sum() / first.sum()

        first_detached = first.detach()
        self.assertAlmostEqual(float(first_x_center.detach()), 0.0, places=5)
        self.assertGreater(
            float(first_detached[20, 20]), float(first_detached[20, 24])
        )
        self.assertEqual(model.recurrent_substeps_per_interval, (1,))
        self.assertEqual(model.q_call_count, 1)
        self.assertEqual(model.source_xy_call_count, 1)

    def test_dense_render_times_use_real_recurrence_and_restore_context(self):
        model = self._ConstantSourceModel()
        configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=[0.0, 1.0],
            u_values=[0.2, 0.2],
            v_values=[0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            nx=41,
            ny=41,
        )
        coarse_times = model.recurrent_times.clone()
        coarse_fields = recurrent_plume_fields(model, sigma_src=0.02).detach()

        dense_fields = recurrent_plume_fields_at_times(
            model,
            sigma_src=0.02,
            t_values=[0.0, 0.5, 1.0],
            u_values=[0.2, 0.2, 0.2],
            v_values=[0.0, 0.0, 0.0],
        ).detach()

        linear_midpoint = 0.5 * (coarse_fields[0] + coarse_fields[1])
        self.assertEqual(tuple(dense_fields.shape), (3, 41, 41))
        self.assertTrue(torch.allclose(dense_fields[0], coarse_fields[0]))
        self.assertFalse(torch.allclose(dense_fields[1], linear_midpoint))
        self.assertTrue(torch.equal(model.recurrent_times, coarse_times))


if __name__ == "__main__":
    unittest.main()
