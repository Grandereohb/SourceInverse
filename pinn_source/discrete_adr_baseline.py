"""Conventional explicit upwind ADR baseline for source-location profiling.

This module is intentionally independent of the differentiable recurrent solver.
It advances cell-centred concentration with first-order upwind advection, centred
diffusion, first-order decay, and a normalized Gaussian source on a padded domain.
The implementation is a transparent numerical baseline, not an operational CFD or
LPDM model.
"""

from __future__ import annotations

import math

import numpy as np

from gaussian_puff_baseline import (
    profile_nonnegative_coefficients,
    profile_nonnegative_scale,
)


def _piecewise_linear_basis(times: np.ndarray, node_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    nodes = np.asarray(node_times, dtype=np.float64).reshape(-1)
    if nodes.size < 2 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("node_times must contain at least two increasing values")
    basis = np.zeros((times.size, nodes.size), dtype=np.float64)
    for index in range(nodes.size):
        values = np.zeros(nodes.size, dtype=np.float64)
        values[index] = 1.0
        basis[:, index] = np.interp(
            times, nodes, values, left=values[0], right=values[-1]
        )
    return basis


def _bilinear_plan(
    x_query: np.ndarray,
    y_query: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> tuple[np.ndarray, ...]:
    gx = (np.asarray(x_query) - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (
        x_grid.size - 1
    )
    gy = (np.asarray(y_query) - y_grid[0]) / (y_grid[-1] - y_grid[0]) * (
        y_grid.size - 1
    )
    if np.any(gx < 0.0) or np.any(gx > x_grid.size - 1):
        raise ValueError("station x coordinate lies outside the transport grid")
    if np.any(gy < 0.0) or np.any(gy > y_grid.size - 1):
        raise ValueError("station y coordinate lies outside the transport grid")
    gx = np.clip(gx, 0.0, x_grid.size - 1)
    gy = np.clip(gy, 0.0, y_grid.size - 1)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = np.minimum(x0 + 1, x_grid.size - 1)
    y1 = np.minimum(y0 + 1, y_grid.size - 1)
    wx = gx - x0
    wy = gy - y0
    return x0, x1, y0, y1, wx, wy


def _sample_fields(fields: np.ndarray, plan: tuple[np.ndarray, ...]) -> np.ndarray:
    x0, x1, y0, y1, wx, wy = plan
    return (
        fields[:, y0, x0] * (1.0 - wx) * (1.0 - wy)
        + fields[:, y0, x1] * wx * (1.0 - wy)
        + fields[:, y1, x0] * (1.0 - wx) * wy
        + fields[:, y1, x1] * wx * wy
    )


def _upwind_adr_step(
    fields: np.ndarray,
    *,
    u_mph: float,
    v_mph: float,
    diffusivity_m2ph: float,
    decay_per_hour: float,
    dt_h: float,
    dx_m: float,
    dy_m: float,
    source_rate: np.ndarray,
    source_kernel: np.ndarray,
) -> np.ndarray:
    padded = np.pad(fields, ((0, 0), (1, 1), (1, 1)), mode="constant")
    centre = padded[:, 1:-1, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    lower = padded[:, :-2, 1:-1]
    upper = padded[:, 2:, 1:-1]
    dcdx = (centre - left) / dx_m if u_mph >= 0.0 else (right - centre) / dx_m
    dcdy = (centre - lower) / dy_m if v_mph >= 0.0 else (upper - centre) / dy_m
    laplacian = (
        (left + right - 2.0 * centre) / dx_m**2
        + (lower + upper - 2.0 * centre) / dy_m**2
    )
    tendency = (
        -u_mph * dcdx
        - v_mph * dcdy
        + diffusivity_m2ph * laplacian
        - decay_per_hour * centre
        + source_rate[:, None, None] * source_kernel[None, :, :]
    )
    return np.maximum(centre + dt_h * tendency, 0.0)


def discrete_adr_station_basis(
    *,
    station_x_m,
    station_y_m,
    observation_times_h,
    wind_times_h,
    wind_u_mps,
    wind_v_mps,
    source_x_m: float,
    source_y_m: float,
    q_node_times_h,
    diffusivity_m2s: float,
    initial_sigma_m: float,
    decay_per_hour: float,
    wind_factor: float,
    transport_bounds_m,
    grid_nx: int = 32,
    grid_ny: int = 32,
    pre_event_hours: float = 1.0,
    cfl_safety: float = 0.75,
) -> tuple[np.ndarray, dict]:
    """Return station responses for unit piecewise-linear source-rate bases."""
    station_x = np.asarray(station_x_m, dtype=np.float64).reshape(-1)
    station_y = np.asarray(station_y_m, dtype=np.float64).reshape(-1)
    observation_times = np.asarray(observation_times_h, dtype=np.float64).reshape(-1)
    wind_times = np.asarray(wind_times_h, dtype=np.float64).reshape(-1)
    wind_u = np.asarray(wind_u_mps, dtype=np.float64).reshape(-1)
    wind_v = np.asarray(wind_v_mps, dtype=np.float64).reshape(-1)
    node_times = np.asarray(q_node_times_h, dtype=np.float64).reshape(-1)
    if observation_times.size == 0 or np.any(np.diff(observation_times) <= 0.0):
        raise ValueError("observation_times_h must be non-empty and increasing")
    if wind_times.size != wind_u.size or wind_times.size != wind_v.size:
        raise ValueError("wind time and component arrays must have equal size")
    if np.any(np.diff(wind_times) < 0.0):
        raise ValueError("wind_times_h must be nondecreasing")
    if diffusivity_m2s < 0.0 or initial_sigma_m <= 0.0 or decay_per_hour < 0.0:
        raise ValueError("invalid transport physics")
    if not (0.0 < cfl_safety < 1.0):
        raise ValueError("cfl_safety must lie between zero and one")
    x_min, x_max, y_min, y_max = map(float, transport_bounds_m)
    grid_nx = max(8, int(grid_nx))
    grid_ny = max(8, int(grid_ny))
    x_grid = np.linspace(x_min, x_max, grid_nx)
    y_grid = np.linspace(y_min, y_max, grid_ny)
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
    source_kernel = np.exp(
        -0.5
        * (
            ((xx - float(source_x_m)) / float(initial_sigma_m)) ** 2
            + ((yy - float(source_y_m)) / float(initial_sigma_m)) ** 2
        )
    )
    source_kernel /= max(float(source_kernel.sum() * dx * dy), 1e-30)
    station_plan = _bilinear_plan(station_x, station_y, x_grid, y_grid)
    fields = np.zeros((node_times.size, grid_ny, grid_nx), dtype=np.float64)
    predictions = np.zeros(
        (observation_times.size, station_x.size, node_times.size), dtype=np.float64
    )
    start_time = float(observation_times[0] - max(pre_event_hours, 0.0))
    targets = observation_times
    current = start_time
    total_substeps = 0
    max_rate = 0.0
    diffusivity_m2ph = float(diffusivity_m2s) * 3600.0

    for output_index, target_time in enumerate(targets):
        while current < target_time - 1e-12:
            u_mph = float(np.interp(current, wind_times, wind_u)) * 3600.0 * wind_factor
            v_mph = float(np.interp(current, wind_times, wind_v)) * 3600.0 * wind_factor
            stability_rate = (
                abs(u_mph) / dx
                + abs(v_mph) / dy
                + 2.0 * diffusivity_m2ph / dx**2
                + 2.0 * diffusivity_m2ph / dy**2
                + float(decay_per_hour)
            )
            max_rate = max(max_rate, stability_rate)
            stable_dt = cfl_safety / max(stability_rate, 1e-12)
            dt = min(stable_dt, target_time - current)
            midpoint = current + 0.5 * dt
            source_rate = _piecewise_linear_basis(
                np.array([midpoint]), node_times
            )[0]
            fields = _upwind_adr_step(
                fields,
                u_mph=u_mph,
                v_mph=v_mph,
                diffusivity_m2ph=diffusivity_m2ph,
                decay_per_hour=float(decay_per_hour),
                dt_h=dt,
                dx_m=dx,
                dy_m=dy,
                source_rate=source_rate,
                source_kernel=source_kernel,
            )
            current += dt
            total_substeps += 1
        predictions[output_index] = _sample_fields(fields, station_plan).T
    audit = {
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "dx_m": dx,
        "dy_m": dy,
        "total_substeps": total_substeps,
        "maximum_stability_rate_per_hour": max_rate,
        "cfl_safety": float(cfl_safety),
        "transport_bounds_m": [x_min, x_max, y_min, y_max],
        "source_kernel_mass": float(source_kernel.sum() * dx * dy),
    }
    return predictions, audit


def search_discrete_adr_profiles(
    *,
    station_x_m,
    station_y_m,
    observation_times_h,
    wind_times_h,
    wind_u_mps,
    wind_v_mps,
    target_anomaly,
    source_bounds_m,
    transport_bounds_m,
    diffusivity_m2s: float,
    initial_sigma_m: float,
    decay_per_hour: float,
    wind_factor: float,
    q_node_count: int = 5,
    grid_size: int = 13,
    refinement_levels: int = 3,
    transport_grid_nx: int = 32,
    transport_grid_ny: int = 32,
    pre_event_hours: float = 1.0,
    return_candidates: bool = False,
) -> list[dict]:
    """Search constant- and dynamic-Q profiles on one discrete ADR surface."""
    if int(q_node_count) < 2 or int(grid_size) < 3 or int(refinement_levels) < 1:
        raise ValueError("invalid profile-search dimensions")
    observation_times = np.asarray(observation_times_h, dtype=np.float64)
    target = np.asarray(target_anomaly, dtype=np.float64)
    expected_shape = (observation_times.size, len(station_x_m))
    if target.shape != expected_shape:
        raise ValueError(f"target_anomaly must have shape {expected_shape}")
    q_node_times = np.linspace(
        float(observation_times[0]),
        float(observation_times[-1]),
        int(q_node_count),
    )
    domain_x_min, domain_x_max, domain_y_min, domain_y_max = map(
        float, source_bounds_m
    )
    x_min, x_max = domain_x_min, domain_x_max
    y_min, y_max = domain_y_min, domain_y_max
    best_constant = None
    best_dynamic = None
    candidates_constant = []
    candidates_dynamic = []
    candidate_count = 0
    equivalent_forward_count = 0
    latest_audit = None

    for level in range(int(refinement_levels)):
        xs = np.linspace(x_min, x_max, int(grid_size))
        ys = np.linspace(y_min, y_max, int(grid_size))
        level_best = None
        for source_y in ys:
            for source_x in xs:
                basis, latest_audit = discrete_adr_station_basis(
                    station_x_m=station_x_m,
                    station_y_m=station_y_m,
                    observation_times_h=observation_times,
                    wind_times_h=wind_times_h,
                    wind_u_mps=wind_u_mps,
                    wind_v_mps=wind_v_mps,
                    source_x_m=float(source_x),
                    source_y_m=float(source_y),
                    q_node_times_h=q_node_times,
                    diffusivity_m2s=diffusivity_m2s,
                    initial_sigma_m=initial_sigma_m,
                    decay_per_hour=decay_per_hour,
                    wind_factor=wind_factor,
                    transport_bounds_m=transport_bounds_m,
                    grid_nx=transport_grid_nx,
                    grid_ny=transport_grid_ny,
                    pre_event_hours=pre_event_hours,
                )
                design = basis.reshape(-1, int(q_node_count))
                constant_unit = design.sum(axis=1)
                amplitude, constant_rmse = profile_nonnegative_scale(
                    constant_unit, target.reshape(-1)
                )
                coefficients, dynamic_rmse = profile_nonnegative_coefficients(
                    design, target.reshape(-1)
                )
                candidate_count += 1
                equivalent_forward_count += int(q_node_count)
                constant = {
                    "source_x_m": float(source_x),
                    "source_y_m": float(source_y),
                    "q_amplitude": amplitude,
                    "anomaly_rmse": constant_rmse,
                    "level": level,
                }
                dynamic = {
                    "source_x_m": float(source_x),
                    "source_y_m": float(source_y),
                    "q_node_values": [float(value) for value in coefficients],
                    "anomaly_rmse": dynamic_rmse,
                    "level": level,
                }
                if return_candidates:
                    candidates_constant.append(dict(constant))
                    candidates_dynamic.append(dict(dynamic))
                if best_constant is None or constant_rmse < best_constant["anomaly_rmse"]:
                    best_constant = constant
                if best_dynamic is None or dynamic_rmse < best_dynamic["anomaly_rmse"]:
                    best_dynamic = dynamic
                if level_best is None or dynamic_rmse < level_best["anomaly_rmse"]:
                    level_best = dynamic
        if level + 1 < int(refinement_levels):
            dx = float(xs[1] - xs[0])
            dy = float(ys[1] - ys[0])
            x_min = max(domain_x_min, level_best["source_x_m"] - dx)
            x_max = min(domain_x_max, level_best["source_x_m"] + dx)
            y_min = max(domain_y_min, level_best["source_y_m"] - dy)
            y_max = min(domain_y_max, level_best["source_y_m"] + dy)

    target_rms = float(math.sqrt(np.mean(target**2)))
    common = {
        "grid_size": int(grid_size),
        "refinement_levels": int(refinement_levels),
        "candidate_count": candidate_count,
        "forward_evaluation_count": equivalent_forward_count,
        "source_bounds_m": list(map(float, source_bounds_m)),
        "transport_audit": latest_audit,
        "physics": {
            "diffusivity_m2s": float(diffusivity_m2s),
            "initial_sigma_m": float(initial_sigma_m),
            "decay_per_hour": float(decay_per_hour),
            "wind_factor": float(wind_factor),
            "pre_event_hours": float(pre_event_hours),
        },
    }
    constant_result = {
        **best_constant,
        **common,
        "method": "B3_constant_Q_discrete_upwind_ADR_profile_grid",
        "normalized_anomaly_rmse": best_constant["anomaly_rmse"]
        / max(target_rms, 1e-12),
    }
    dynamic_result = {
        **best_dynamic,
        **common,
        "method": "B3_dynamic_Q_discrete_upwind_ADR_profile_grid",
        "q_parameterization": "nonnegative_piecewise_linear_nodes",
        "q_node_times_h": [float(value) for value in q_node_times],
        "normalized_anomaly_rmse": best_dynamic["anomaly_rmse"]
        / max(target_rms, 1e-12),
    }
    if return_candidates:
        constant_result["candidates"] = candidates_constant
        dynamic_result["candidates"] = candidates_dynamic
    return [constant_result, dynamic_result]
