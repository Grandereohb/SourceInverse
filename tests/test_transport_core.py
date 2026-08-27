import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import field as transport  # noqa: E402
from characteristic_source import (  # noqa: E402
    advance_interval_characteristic_source,
    source_aware_quadrature_count,
)
from research_recurrent import recurrent_plume_fields_characteristic  # noqa: E402
from transport_units import (  # noqa: E402
    normalize_decay_per_hour,
    normalize_diffusivity_m2s,
    normalize_velocity_mps,
)


def _grid(n=81, extent=1.5, dtype=torch.float64):
    values = torch.linspace(-extent, extent, n, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return values, xx, yy


def _integral(field, grid):
    dx = grid[1] - grid[0]
    return torch.sum(field) * dx * dx


def _normalized_gaussian(xx, yy, x0, y0, sigma, grid):
    field = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))
    return field / _integral(field, grid)


class _DifferentiableSourceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.xs = torch.nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))
        self.ys = torch.nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.raw_d = torch.nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        self.log_q = torch.nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

    def D(self):
        return torch.nn.functional.softplus(self.raw_d)

    def Q(self, t):
        return torch.exp(self.log_q) * torch.ones_like(t)

    def source_xy(self, t):
        return self.xs.expand_as(t), self.ys.expand_as(t)


class TransportUnitTests(unittest.TestCase):
    def test_physical_unit_normalization(self):
        velocity = normalize_velocity_mps(1.0, 2.0, 7200.0, factor=0.25)
        diffusion = normalize_diffusivity_m2s(2.0, 2.0, 7200.0)
        decay = normalize_decay_per_hour(0.05, 12.0)

        self.assertAlmostEqual(float(velocity), 0.25, places=7)
        self.assertAlmostEqual(float(diffusion), 1.0 / 3600.0, places=10)
        self.assertAlmostEqual(decay, 0.6, places=7)

    def test_adaptive_substeps_report_required_count_and_cap(self):
        selected, required, distances = transport._build_adaptive_substep_plan(
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
        self.assertAlmostEqual(distances[0], 20.0)


class AdvectionDiffusionDecayTests(unittest.TestCase):
    def test_smooth_gaussian_advection_moves_in_wind_direction(self):
        grid, xx, yy = _grid(n=121, extent=1.5)
        initial = _normalized_gaussian(xx, yy, -0.25, 0.20, 0.16, grid)
        u = torch.tensor(0.20, dtype=grid.dtype)
        v = torch.tensor(-0.10, dtype=grid.dtype)
        dt = torch.tensor(0.50, dtype=grid.dtype)

        moved = transport._advect_field(
            initial,
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            u,
            v,
            dt,
        )
        mass = _integral(moved, grid)
        x_center = _integral(moved * xx, grid) / mass
        y_center = _integral(moved * yy, grid) / mass

        self.assertAlmostEqual(float(mass), 1.0, delta=2e-5)
        self.assertAlmostEqual(float(x_center), -0.15, delta=2e-4)
        self.assertAlmostEqual(float(y_center), 0.15, delta=2e-4)

    def test_smooth_advection_error_decreases_with_grid_refinement(self):
        errors = []
        for n in (41, 81, 161):
            grid, xx, yy = _grid(n=n, extent=1.5)
            initial = _normalized_gaussian(xx, yy, -0.30, 0.15, 0.17, grid)
            u = torch.tensor(0.23, dtype=grid.dtype)
            v = torch.tensor(-0.11, dtype=grid.dtype)
            dt = torch.tensor(0.47, dtype=grid.dtype)
            moved = transport._advect_field(
                initial,
                grid,
                grid,
                xx.reshape(-1),
                yy.reshape(-1),
                u,
                v,
                dt,
            )
            expected = _normalized_gaussian(
                xx,
                yy,
                -0.30 + float(u * dt),
                0.15 + float(v * dt),
                0.17,
                grid,
            )
            relative_l2 = torch.linalg.vector_norm(moved - expected) / torch.linalg.vector_norm(
                expected
            )
            errors.append(float(relative_l2))

        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 0.015)

    def test_gaussian_diffusion_matches_mass_and_variance(self):
        grid, xx, yy = _grid(n=121, extent=1.5)
        sigma0 = 0.12
        diffusion = torch.tensor(0.006, dtype=grid.dtype)
        dt = torch.tensor(0.40, dtype=grid.dtype)
        initial = _normalized_gaussian(xx, yy, 0.0, 0.0, sigma0, grid)

        diffused = transport._diffuse_field(initial, grid, grid, diffusion, dt)
        mass = _integral(diffused, grid)
        radial_variance = _integral(diffused * (xx**2 + yy**2), grid) / mass
        expected_radial_variance = 2.0 * (sigma0**2 + 2.0 * float(diffusion * dt))

        self.assertAlmostEqual(float(mass), 1.0, delta=2e-5)
        self.assertAlmostEqual(
            float(radial_variance), expected_radial_variance, delta=8e-4
        )

    def test_combined_adr_matches_gaussian_reference(self):
        grid, xx, yy = _grid(n=161, extent=2.0)
        x0, y0, sigma0 = -0.35, 0.20, 0.18
        u = torch.tensor(0.16, dtype=grid.dtype)
        v = torch.tensor(-0.07, dtype=grid.dtype)
        diffusion = torch.tensor(0.004, dtype=grid.dtype)
        decay = 0.30
        dt = torch.tensor(0.50, dtype=grid.dtype)
        initial = _normalized_gaussian(xx, yy, x0, y0, sigma0, grid)
        zero_source = torch.zeros_like(initial)

        numerical = transport._advance_recurrent_step(
            initial,
            zero_source,
            torch.tensor(0.0, dtype=grid.dtype),
            zero_source,
            torch.tensor(0.0, dtype=grid.dtype),
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            u,
            v,
            diffusion,
            decay,
            1.0,
            dt,
        )
        sigma_t = math.sqrt(sigma0**2 + 2.0 * float(diffusion * dt))
        expected = _normalized_gaussian(
            xx,
            yy,
            x0 + float(u * dt),
            y0 + float(v * dt),
            sigma_t,
            grid,
        ) * math.exp(-decay * float(dt))
        relative_l2 = torch.linalg.vector_norm(numerical - expected) / torch.linalg.vector_norm(
            expected
        )

        self.assertLess(float(relative_l2), 0.025)
        self.assertAlmostEqual(
            float(_integral(numerical, grid)), math.exp(-decay * float(dt)), delta=3e-4
        )

    def test_open_boundary_loses_outflow_mass(self):
        grid, xx, yy = _grid(n=81, extent=1.0)
        initial = _normalized_gaussian(xx, yy, 0.86, 0.0, 0.05, grid)
        moved = transport._advect_field(
            initial,
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            torch.tensor(0.35, dtype=grid.dtype),
            torch.tensor(0.0, dtype=grid.dtype),
            torch.tensor(1.0, dtype=grid.dtype),
        )

        self.assertLess(float(_integral(moved, grid)), 0.01)


class SourceIntegrationTests(unittest.TestCase):
    def test_symmetric_endpoint_source_mass(self):
        grid, xx, yy = _grid(n=81, extent=1.5)
        source = _normalized_gaussian(xx, yy, 0.0, 0.0, 0.08, grid)
        zero = torch.zeros_like(source)
        dt = torch.tensor(0.50, dtype=grid.dtype)

        advanced = transport._advance_recurrent_step(
            zero,
            source,
            torch.tensor(2.0, dtype=grid.dtype),
            source,
            torch.tensor(4.0, dtype=grid.dtype),
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            torch.tensor(0.0, dtype=grid.dtype),
            torch.tensor(0.0, dtype=grid.dtype),
            torch.tensor(1e-12, dtype=grid.dtype),
            0.0,
            1.0,
            dt,
        )

        expected_mass = 0.5 * (2.0 + 4.0) * float(dt)
        self.assertAlmostEqual(float(_integral(advanced, grid)), expected_mass, delta=2e-5)

    def test_initial_release_fraction_controls_first_stored_mass(self):
        model = _DifferentiableSourceModel()
        transport.configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=[0.0, 0.75],
            u_values=[0.0, 0.0],
            v_values=[0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            initial_release_dt=0.75,
            nx=81,
            ny=81,
        )

        with patch.object(transport, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.0):
            no_preload = transport.recurrent_plume_fields(model, sigma_src=0.05)[0]
        with patch.object(transport, "RECURRENT_INITIAL_RELEASE_FRACTION", 1.0):
            current_preload = transport.recurrent_plume_fields(model, sigma_src=0.05)[0]

        grid = model.recurrent_x_grid
        expected = math.exp(float(model.log_q.detach())) * 0.75
        self.assertAlmostEqual(
            float(_integral(no_preload, grid).detach()), 0.0, delta=1e-8
        )
        self.assertAlmostEqual(
            float(_integral(current_preload, grid).detach()), expected, delta=2e-4
        )

    def test_single_observation_uses_declared_initial_release_rule(self):
        model = _DifferentiableSourceModel()
        transport.configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=[0.0],
            u_values=[0.0],
            v_values=[0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            initial_release_dt=0.25,
            nx=81,
            ny=81,
        )

        with patch.object(transport, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.0):
            no_preload = transport.recurrent_plume_fields(model, sigma_src=0.05)[0]
        with patch.object(transport, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.5):
            half_preload = transport.recurrent_plume_fields(model, sigma_src=0.05)[0]

        grid = model.recurrent_x_grid
        expected_half = math.exp(float(model.log_q.detach())) * 0.25 * 0.5
        self.assertAlmostEqual(
            float(_integral(no_preload, grid).detach()), 0.0, delta=1e-8
        )
        self.assertAlmostEqual(
            float(_integral(half_preload, grid).detach()), expected_half, delta=2e-4
        )

    def test_irregular_intervals_follow_cumulative_source_mass(self):
        model = _DifferentiableSourceModel()
        times = [0.0, 0.25, 1.0]
        transport.configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=times,
            u_values=[0.0, 0.0, 0.0],
            v_values=[0.0, 0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            nx=81,
            ny=81,
        )

        with patch.object(transport, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.0):
            fields = transport.recurrent_plume_fields(model, sigma_src=0.05)

        grid = model.recurrent_x_grid
        q_value = math.exp(float(model.log_q.detach()))
        observed_mass = [float(_integral(field, grid).detach()) for field in fields]
        expected_mass = [q_value * (time - times[0]) for time in times]
        for observed, expected in zip(observed_mass, expected_mass):
            self.assertAlmostEqual(observed, expected, delta=4e-4)


class GradientVerificationTests(unittest.TestCase):
    def _configure(self, model):
        transport.configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=[0.0, 0.4, 0.8],
            u_values=[0.14, 0.14, 0.14],
            v_values=[-0.04, -0.04, -0.04],
            d_min_norm=0.0,
            d_scale_norm=1.0,
            decay_norm=0.08,
            nx=61,
            ny=61,
        )

    def _objective(self, model):
        with patch.object(transport, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.0):
            final_field = transport.recurrent_plume_fields(model, sigma_src=0.06)[-1]
        grid = model.recurrent_x_grid
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        probe = torch.exp(-((xx - 0.24) ** 2 + (yy + 0.03) ** 2) / (2.0 * 0.11**2))
        return _integral(final_field * probe, grid)

    def _central_difference(self, model, parameter, step):
        original = float(parameter.detach())
        with torch.no_grad():
            parameter.fill_(original + step)
        plus = float(self._objective(model).detach())
        with torch.no_grad():
            parameter.fill_(original - step)
        minus = float(self._objective(model).detach())
        with torch.no_grad():
            parameter.fill_(original)
        return (plus - minus) / (2.0 * step)

    def test_source_diffusion_and_release_gradients_match_finite_differences(self):
        model = _DifferentiableSourceModel()
        self._configure(model)
        objective = self._objective(model)
        parameters = (model.xs, model.ys, model.raw_d, model.log_q)
        automatic = torch.autograd.grad(objective, parameters)
        steps = (2e-3, 2e-3, 2e-3, 2e-3)

        for name, parameter, auto, step in zip(
            ("x_s", "y_s", "raw_D", "log_Q"), parameters, automatic, steps
        ):
            finite = self._central_difference(model, parameter, step)
            scale = max(abs(finite), abs(float(auto)), 1e-6)
            relative_error = abs(float(auto) - finite) / scale
            self.assertLess(
                relative_error,
                0.03,
                msg=(
                    f"{name}: autograd={float(auto):.8g}, finite={finite:.8g}, "
                    f"relative_error={relative_error:.3g}"
                ),
            )


class CharacteristicSourceQuadratureTests(unittest.TestCase):
    def _continuous_release_reference(
        self, grid, xx, yy, source_x, sigma, velocity, samples=2001
    ):
        base = torch.exp(-((xx - source_x) ** 2 + yy**2) / (2.0 * sigma**2))
        source_mass = _integral(base, grid)
        release_time = torch.linspace(0.0, 1.0, samples, dtype=grid.dtype)
        age = (1.0 - release_time).view(-1, 1, 1)
        x_back = xx.view(1, *xx.shape) - velocity * age
        within = (x_back >= grid[0]) & (x_back <= grid[-1])
        values = torch.where(
            within,
            torch.exp(
                -(
                    (x_back - source_x) ** 2
                    + yy.view(1, *yy.shape) ** 2
                )
                / (2.0 * sigma**2)
            )
            / source_mass,
            0.0,
        )
        values[0] *= 0.5
        values[-1] *= 0.5
        return values.sum(dim=0) / (samples - 1)

    def test_source_aware_count_uses_travel_relative_to_kernel_width(self):
        selected, requested = source_aware_quadrature_count(
            u=0.8,
            v=0.0,
            dt=1.0,
            source_sigma=0.05,
            max_source_sigmas_per_panel=1.0,
            maximum_panels=128,
        )
        self.assertEqual(requested, 16)
        self.assertEqual(selected, 16)

    def test_characteristic_quadrature_resolves_high_displacement_release(self):
        grid = torch.linspace(-0.5, 0.5, 36, dtype=torch.float64)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        source_x = -0.4
        sigma = 0.05
        source = _normalized_gaussian(xx, yy, source_x, 0.0, sigma, grid)
        reference = self._continuous_release_reference(
            grid, xx, yy, source_x, sigma, velocity=0.8
        )
        panels, _ = source_aware_quadrature_count(
            u=0.8,
            v=0.0,
            dt=1.0,
            source_sigma=sigma,
            max_source_sigmas_per_panel=1.0,
        )
        characteristic = advance_interval_characteristic_source(
            field=torch.zeros_like(source),
            source=source,
            q_samples=torch.ones(panels + 1, dtype=grid.dtype),
            x_grid=grid,
            y_grid=grid,
            x_mesh_flat=xx.reshape(-1),
            y_mesh_flat=yy.reshape(-1),
            u=torch.tensor(0.8, dtype=grid.dtype),
            v=torch.tensor(0.0, dtype=grid.dtype),
            diffusion=torch.tensor(0.0, dtype=grid.dtype),
            decay=0.0,
            source_scale=1.0,
            dt=torch.tensor(1.0, dtype=grid.dtype),
        )
        relative_l2 = torch.linalg.vector_norm(
            characteristic - reference
        ) / torch.linalg.vector_norm(reference)

        self.assertLess(float(relative_l2), 0.04)

    def test_characteristic_quadrature_matches_open_boundary_reference(self):
        grid = torch.linspace(-0.5, 0.5, 36, dtype=torch.float64)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        sigma = 0.05
        source = _normalized_gaussian(xx, yy, 0.0, 0.0, sigma, grid)
        reference = self._continuous_release_reference(
            grid, xx, yy, source_x=0.0, sigma=sigma, velocity=0.8
        )
        panels, _ = source_aware_quadrature_count(
            u=0.8,
            v=0.0,
            dt=1.0,
            source_sigma=sigma,
            max_source_sigmas_per_panel=1.0,
        )
        characteristic = advance_interval_characteristic_source(
            field=torch.zeros_like(source),
            source=source,
            q_samples=torch.ones(panels + 1, dtype=grid.dtype),
            x_grid=grid,
            y_grid=grid,
            x_mesh_flat=xx.reshape(-1),
            y_mesh_flat=yy.reshape(-1),
            u=torch.tensor(0.8, dtype=grid.dtype),
            v=torch.tensor(0.0, dtype=grid.dtype),
            diffusion=torch.tensor(0.0, dtype=grid.dtype),
            decay=0.0,
            source_scale=1.0,
            dt=torch.tensor(1.0, dtype=grid.dtype),
        )
        relative_l2 = torch.linalg.vector_norm(
            characteristic - reference
        ) / torch.linalg.vector_norm(reference)
        characteristic_mass = _integral(characteristic, grid)
        reference_mass = _integral(reference, grid)

        self.assertLess(float(relative_l2), 0.025)
        self.assertAlmostEqual(
            float(characteristic_mass), float(reference_mass), delta=5e-4
        )
        self.assertLess(float(characteristic_mass), 0.7)

    def test_characteristic_quadrature_preserves_zero_transport_source_mass(self):
        grid, xx, yy = _grid(n=61, extent=1.0)
        source = _normalized_gaussian(xx, yy, 0.0, 0.0, 0.08, grid)
        q_samples = torch.linspace(2.0, 4.0, 9, dtype=grid.dtype)
        result = advance_interval_characteristic_source(
            field=torch.zeros_like(source),
            source=source,
            q_samples=q_samples,
            x_grid=grid,
            y_grid=grid,
            x_mesh_flat=xx.reshape(-1),
            y_mesh_flat=yy.reshape(-1),
            u=torch.tensor(0.0, dtype=grid.dtype),
            v=torch.tensor(0.0, dtype=grid.dtype),
            diffusion=torch.tensor(0.0, dtype=grid.dtype),
            decay=0.0,
            source_scale=1.0,
            dt=torch.tensor(0.5, dtype=grid.dtype),
        )
        expected_mass = 0.5 * 0.5 * (2.0 + 4.0)
        self.assertAlmostEqual(float(_integral(result, grid)), expected_mass, delta=2e-5)

    def test_characteristic_source_gradients_match_finite_differences(self):
        grid = torch.linspace(-0.6, 0.6, 51, dtype=torch.float64)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        xs = torch.nn.Parameter(torch.tensor(-0.18, dtype=grid.dtype))
        ys = torch.nn.Parameter(torch.tensor(0.09, dtype=grid.dtype))
        raw_d = torch.nn.Parameter(torch.tensor(-5.0, dtype=grid.dtype))
        log_q_start = torch.nn.Parameter(torch.tensor(0.15, dtype=grid.dtype))
        log_q_end = torch.nn.Parameter(torch.tensor(-0.10, dtype=grid.dtype))
        parameters = (xs, ys, raw_d, log_q_start, log_q_end)
        panels = 6

        def objective():
            source = torch.exp(
                -((xx - xs) ** 2 + (yy - ys) ** 2) / (2.0 * 0.055**2)
            )
            source = source / _integral(source, grid)
            fraction = torch.linspace(0.0, 1.0, panels + 1, dtype=grid.dtype)
            q_samples = torch.exp(
                (1.0 - fraction) * log_q_start + fraction * log_q_end
            )
            result = advance_interval_characteristic_source(
                field=torch.zeros_like(source),
                source=source,
                q_samples=q_samples,
                x_grid=grid,
                y_grid=grid,
                x_mesh_flat=xx.reshape(-1),
                y_mesh_flat=yy.reshape(-1),
                u=torch.tensor(0.30, dtype=grid.dtype),
                v=torch.tensor(-0.05, dtype=grid.dtype),
                diffusion=torch.nn.functional.softplus(raw_d),
                decay=0.12,
                source_scale=1.0,
                dt=torch.tensor(0.50, dtype=grid.dtype),
            )
            probe = torch.exp(
                -((xx - 0.12) ** 2 + (yy + 0.02) ** 2) / (2.0 * 0.10**2)
            )
            return _integral(result * probe, grid)

        automatic = torch.autograd.grad(objective(), parameters)
        step = 1e-3
        for name, parameter, auto in zip(
            ("x_s", "y_s", "raw_D", "log_Q_start", "log_Q_end"),
            parameters,
            automatic,
        ):
            original = float(parameter.detach())
            with torch.no_grad():
                parameter.fill_(original + step)
            plus = float(objective().detach())
            with torch.no_grad():
                parameter.fill_(original - step)
            minus = float(objective().detach())
            with torch.no_grad():
                parameter.fill_(original)
            finite = (plus - minus) / (2.0 * step)
            scale = max(abs(finite), abs(float(auto)), 1e-7)
            relative_error = abs(float(auto) - finite) / scale
            self.assertLess(
                relative_error,
                0.03,
                msg=(
                    f"{name}: autograd={float(auto):.8g}, finite={finite:.8g}, "
                    f"relative_error={relative_error:.3g}"
                ),
            )

    def test_characteristic_float32_agrees_with_float64(self):
        def run(dtype):
            grid = torch.linspace(-0.6, 0.6, 61, dtype=dtype)
            yy, xx = torch.meshgrid(grid, grid, indexing="ij")
            source = _normalized_gaussian(xx, yy, -0.2, 0.07, 0.06, grid)
            return advance_interval_characteristic_source(
                field=torch.zeros_like(source),
                source=source,
                q_samples=torch.linspace(1.1, 0.7, 9, dtype=dtype),
                x_grid=grid,
                y_grid=grid,
                x_mesh_flat=xx.reshape(-1),
                y_mesh_flat=yy.reshape(-1),
                u=torch.tensor(0.26, dtype=dtype),
                v=torch.tensor(-0.08, dtype=dtype),
                diffusion=torch.tensor(0.004, dtype=dtype),
                decay=0.15,
                source_scale=1.0,
                dt=torch.tensor(0.6, dtype=dtype),
            )

        result64 = run(torch.float64)
        result32 = run(torch.float32).to(torch.float64)
        relative_l2 = torch.linalg.vector_norm(
            result32 - result64
        ) / torch.linalg.vector_norm(result64)
        self.assertLess(float(relative_l2), 2e-4)


class ResearchRecurrentTests(unittest.TestCase):
    class _LinearQModel(_DifferentiableSourceModel):
        def Q(self, t):
            return 1.0 + 2.0 * t

    def test_solver_dispatch_matches_explicit_implementations(self):
        model = _DifferentiableSourceModel()
        transport.configure_recurrent_context(
            model,
            x_min=-0.5,
            x_max=0.5,
            y_min=-0.5,
            y_max=0.5,
            t_values=[0.0, 0.5, 1.0],
            u_values=[0.2, 0.2, 0.2],
            v_values=[0.0, 0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=1.0,
            decay_norm=0.1,
            nx=36,
            ny=36,
        )

        model.recurrent_solver = "production"
        selected_production = transport.recurrent_plume_fields_selected(model, 0.05)
        explicit_production = transport.recurrent_plume_fields(model, 0.05)
        torch.testing.assert_close(selected_production, explicit_production)

        model.recurrent_solver = "characteristic"
        selected_characteristic = transport.recurrent_plume_fields_selected(model, 0.05)
        explicit_characteristic = recurrent_plume_fields_characteristic(model, 0.05)
        torch.testing.assert_close(selected_characteristic, explicit_characteristic)

    def test_solver_dispatch_rejects_unknown_name(self):
        model = _DifferentiableSourceModel()
        model.recurrent_solver = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown recurrent solver"):
            transport.recurrent_plume_fields_selected(model, 0.05)

    def test_irregular_intervals_integrate_linear_q_mass(self):
        model = self._LinearQModel()
        times = [0.0, 0.2, 0.7, 1.0]
        transport.configure_recurrent_context(
            model,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            t_values=times,
            u_values=[0.0] * len(times),
            v_values=[0.0] * len(times),
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            nx=61,
            ny=61,
        )
        fields = recurrent_plume_fields_characteristic(
            model, sigma_src=0.06, initial_release_fraction=0.0
        )
        grid = model.recurrent_x_grid
        masses = [float(_integral(field, grid).detach()) for field in fields]
        expected = [time + time**2 for time in times]
        for observed, target in zip(masses, expected):
            self.assertAlmostEqual(observed, target, delta=8e-4)

    def test_multi_interval_high_displacement_matches_characteristic_reference(self):
        model = _DifferentiableSourceModel()
        with torch.no_grad():
            model.xs.fill_(-0.4)
            model.ys.fill_(0.0)
            model.log_q.fill_(0.0)
        transport.configure_recurrent_context(
            model,
            x_min=-0.5,
            x_max=0.5,
            y_min=-0.5,
            y_max=0.5,
            t_values=[0.0, 0.5, 1.0],
            u_values=[0.8, 0.8, 0.8],
            v_values=[0.0, 0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            nx=36,
            ny=36,
        )
        fields = recurrent_plume_fields_characteristic(
            model,
            sigma_src=0.05,
            initial_release_fraction=0.0,
            max_source_sigmas_per_panel=1.0,
        )
        grid = model.recurrent_x_grid.to(torch.float64)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        base = torch.exp(-((xx + 0.4) ** 2 + yy**2) / (2.0 * 0.05**2))
        source_mass = _integral(base, grid)
        release_time = torch.linspace(0.0, 1.0, 2001, dtype=grid.dtype)
        age = (1.0 - release_time).view(-1, 1, 1)
        x_back = xx.view(1, *xx.shape) - 0.8 * age
        within = (x_back >= grid[0]) & (x_back <= grid[-1])
        reference_samples = torch.where(
            within,
            torch.exp(-((x_back + 0.4) ** 2 + yy.view(1, *yy.shape) ** 2) / (2.0 * 0.05**2))
            / source_mass,
            0.0,
        )
        reference_samples[0] *= 0.5
        reference_samples[-1] *= 0.5
        reference = reference_samples.sum(dim=0) / (release_time.numel() - 1)
        relative_l2 = torch.linalg.vector_norm(
            fields[-1].to(torch.float64) - reference
        ) / torch.linalg.vector_norm(reference)

        self.assertEqual(model.research_source_quadrature_panels_per_interval, (8, 8))
        self.assertLess(float(relative_l2.detach()), 0.05)

    def test_research_recurrent_records_quadrature_cap_hits(self):
        model = _DifferentiableSourceModel()
        transport.configure_recurrent_context(
            model,
            x_min=-0.5,
            x_max=0.5,
            y_min=-0.5,
            y_max=0.5,
            t_values=[0.0, 1.0],
            u_values=[0.8, 0.8],
            v_values=[0.0, 0.0],
            d_min_norm=0.0,
            d_scale_norm=0.0,
            decay_norm=0.0,
            nx=36,
            ny=36,
        )
        recurrent_plume_fields_characteristic(
            model,
            sigma_src=0.05,
            initial_release_fraction=0.0,
            maximum_source_panels=4,
        )
        self.assertEqual(model.research_source_quadrature_panels_per_interval, (4,))
        self.assertEqual(model.research_source_quadrature_required_per_interval, (16,))
        self.assertEqual(model.research_source_quadrature_cap_hit_count, 1)

    def test_research_recurrent_gradients_match_finite_differences(self):
        model = _DifferentiableSourceModel()
        transport.configure_recurrent_context(
            model,
            x_min=-0.7,
            x_max=0.7,
            y_min=-0.7,
            y_max=0.7,
            t_values=[0.0, 0.35, 0.8],
            u_values=[0.20, 0.16, 0.16],
            v_values=[-0.03, 0.04, 0.04],
            d_min_norm=0.0,
            d_scale_norm=1.0,
            decay_norm=0.10,
            nx=51,
            ny=51,
        )

        def objective():
            final = recurrent_plume_fields_characteristic(
                model,
                sigma_src=0.06,
                initial_release_fraction=0.0,
                max_source_sigmas_per_panel=1.0,
            )[-1]
            grid = model.recurrent_x_grid
            yy, xx = torch.meshgrid(grid, grid, indexing="ij")
            probe = torch.exp(
                -((xx - 0.18) ** 2 + (yy + 0.04) ** 2) / (2.0 * 0.11**2)
            )
            return _integral(final * probe, grid)

        parameters = (model.xs, model.ys, model.raw_d, model.log_q)
        automatic = torch.autograd.grad(objective(), parameters)
        step = 2e-3
        for name, parameter, auto in zip(
            ("x_s", "y_s", "raw_D", "log_Q"), parameters, automatic
        ):
            original = float(parameter.detach())
            with torch.no_grad():
                parameter.fill_(original + step)
            plus = float(objective().detach())
            with torch.no_grad():
                parameter.fill_(original - step)
            minus = float(objective().detach())
            with torch.no_grad():
                parameter.fill_(original)
            finite = (plus - minus) / (2.0 * step)
            scale = max(abs(finite), abs(float(auto)), 1e-6)
            relative_error = abs(float(auto) - finite) / scale
            self.assertLess(
                relative_error,
                0.04,
                msg=(
                    f"{name}: autograd={float(auto):.8g}, finite={finite:.8g}, "
                    f"relative_error={relative_error:.3g}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
