"""Deterministic Gaussian-puff baseline with profiled constant source strength."""

from __future__ import annotations

import math

import numpy as np

from synthetic_puff import gaussian_puff_station_concentrations


def profile_nonnegative_scale(unit_prediction, target) -> tuple[float, float]:
    prediction = np.asarray(unit_prediction, dtype=np.float64).reshape(-1)
    observed = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.size != observed.size or prediction.size == 0:
        raise ValueError("prediction and target must have the same non-zero size")
    denominator = float(np.dot(prediction, prediction))
    amplitude = max(float(np.dot(prediction, observed)) / max(denominator, 1e-30), 0.0)
    residual = amplitude * prediction - observed
    return amplitude, float(math.sqrt(np.mean(residual**2)))


def profile_nonnegative_coefficients(
    design_matrix, target, *, iterations=250
) -> tuple[np.ndarray, float]:
    design = np.asarray(design_matrix, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape[0] != observed.size or design.shape[1] == 0:
        raise ValueError("design_matrix rows must match a non-empty target")
    coefficients = np.maximum(np.linalg.lstsq(design, observed, rcond=None)[0], 0.0)
    lipschitz = float(np.linalg.norm(design, ord=2) ** 2)
    if lipschitz > 0.0:
        step = 1.0 / lipschitz
        for _ in range(max(int(iterations), 1)):
            gradient = design.T @ (design @ coefficients - observed)
            updated = np.maximum(coefficients - step * gradient, 0.0)
            if np.linalg.norm(updated - coefficients) <= 1e-10 * (
                1.0 + np.linalg.norm(coefficients)
            ):
                coefficients = updated
                break
            coefficients = updated
    residual = design @ coefficients - observed
    return coefficients, float(math.sqrt(np.mean(residual**2)))


def search_constant_q_gaussian_puff(
    *,
    station_x_m,
    station_y_m,
    observation_times_h,
    wind_times_h,
    wind_u_mps,
    wind_v_mps,
    target_anomaly,
    source_bounds_m,
    diffusivity_m2s,
    initial_sigma_m,
    decay_per_hour,
    wind_factor,
    pre_event_hours=1.0,
    integration_dt_h=0.05,
    grid_size=17,
    refinement_levels=3,
    return_candidates=False,
) -> dict:
    if int(grid_size) < 3 or int(refinement_levels) < 1:
        raise ValueError("grid_size >= 3 and refinement_levels >= 1 are required")
    target = np.asarray(target_anomaly, dtype=np.float64)
    expected_shape = (len(observation_times_h), len(station_x_m))
    if target.shape != expected_shape:
        raise ValueError(f"target_anomaly must have shape {expected_shape}")
    x_domain_min, x_domain_max, y_domain_min, y_domain_max = map(
        float, source_bounds_m
    )
    if x_domain_min >= x_domain_max or y_domain_min >= y_domain_max:
        raise ValueError("invalid source bounds")
    x_min, x_max = x_domain_min, x_domain_max
    y_min, y_max = y_domain_min, y_domain_max
    best = None
    evaluation_count = 0
    candidates = []

    def constant_q(times):
        return np.ones_like(np.asarray(times, dtype=np.float64))

    for level in range(int(refinement_levels)):
        xs = np.linspace(x_min, x_max, int(grid_size))
        ys = np.linspace(y_min, y_max, int(grid_size))
        for source_y in ys:
            for source_x in xs:
                unit = gaussian_puff_station_concentrations(
                    station_x_m=station_x_m,
                    station_y_m=station_y_m,
                    observation_times_h=observation_times_h,
                    wind_times_h=wind_times_h,
                    wind_u_mps=wind_u_mps,
                    wind_v_mps=wind_v_mps,
                    source_x_m=float(source_x),
                    source_y_m=float(source_y),
                    q_at_time=constant_q,
                    diffusivity_m2s=diffusivity_m2s,
                    initial_sigma_m=initial_sigma_m,
                    decay_per_hour=decay_per_hour,
                    wind_factor=wind_factor,
                    pre_event_hours=pre_event_hours,
                    integration_dt_h=integration_dt_h,
                )
                amplitude, rmse = profile_nonnegative_scale(unit, target)
                evaluation_count += 1
                candidate = {
                    "source_x_m": float(source_x),
                    "source_y_m": float(source_y),
                    "q_amplitude": amplitude,
                    "anomaly_rmse": rmse,
                    "level": level,
                }
                if return_candidates:
                    candidates.append(dict(candidate))
                if best is None or rmse < best["anomaly_rmse"]:
                    best = candidate
        if level + 1 < int(refinement_levels):
            dx = float(xs[1] - xs[0])
            dy = float(ys[1] - ys[0])
            x_min = max(x_domain_min, best["source_x_m"] - dx)
            x_max = min(x_domain_max, best["source_x_m"] + dx)
            y_min = max(y_domain_min, best["source_y_m"] - dy)
            y_max = min(y_domain_max, best["source_y_m"] + dy)
    target_rms = float(math.sqrt(np.mean(target**2)))
    result = {
        **best,
        "method": "B2_constant_Q_gaussian_puff_profile_grid",
        "normalized_anomaly_rmse": best["anomaly_rmse"] / max(target_rms, 1e-12),
        "grid_size": int(grid_size),
        "refinement_levels": int(refinement_levels),
        "forward_evaluation_count": evaluation_count,
        "source_bounds_m": [
            x_domain_min,
            x_domain_max,
            y_domain_min,
            y_domain_max,
        ],
        "physics": {
            "diffusivity_m2s": float(diffusivity_m2s),
            "initial_sigma_m": float(initial_sigma_m),
            "decay_per_hour": float(decay_per_hour),
            "wind_factor": float(wind_factor),
            "pre_event_hours": float(pre_event_hours),
            "integration_dt_h": float(integration_dt_h),
        },
    }
    if return_candidates:
        result["candidates"] = candidates
    return result


def search_dynamic_q_gaussian_puff(
    *,
    station_x_m,
    station_y_m,
    observation_times_h,
    wind_times_h,
    wind_u_mps,
    wind_v_mps,
    target_anomaly,
    source_bounds_m,
    diffusivity_m2s,
    initial_sigma_m,
    decay_per_hour,
    wind_factor,
    pre_event_hours=1.0,
    integration_dt_h=0.05,
    q_node_count=5,
    nnls_iterations=250,
    grid_size=13,
    refinement_levels=3,
    return_candidates=False,
) -> dict:
    if int(q_node_count) < 2:
        raise ValueError("q_node_count must be at least 2")
    if int(grid_size) < 3 or int(refinement_levels) < 1:
        raise ValueError("grid_size >= 3 and refinement_levels >= 1 are required")
    observation_times = np.asarray(observation_times_h, dtype=np.float64)
    target = np.asarray(target_anomaly, dtype=np.float64)
    expected_shape = (observation_times.size, len(station_x_m))
    if target.shape != expected_shape:
        raise ValueError(f"target_anomaly must have shape {expected_shape}")
    node_times = np.linspace(
        float(observation_times[0]),
        float(observation_times[-1]),
        int(q_node_count),
    )
    x_domain_min, x_domain_max, y_domain_min, y_domain_max = map(
        float, source_bounds_m
    )
    if x_domain_min >= x_domain_max or y_domain_min >= y_domain_max:
        raise ValueError("invalid source bounds")
    x_min, x_max = x_domain_min, x_domain_max
    y_min, y_max = y_domain_min, y_domain_max
    best = None
    evaluation_count = 0
    candidates = []

    def basis_q(node_index):
        node_values = np.zeros(int(q_node_count), dtype=np.float64)
        node_values[node_index] = 1.0

        def q_at_time(times):
            return np.interp(
                np.asarray(times, dtype=np.float64),
                node_times,
                node_values,
                left=float(node_values[0]),
                right=float(node_values[-1]),
            )

        return q_at_time

    q_basis_functions = [basis_q(index) for index in range(int(q_node_count))]
    for level in range(int(refinement_levels)):
        xs = np.linspace(x_min, x_max, int(grid_size))
        ys = np.linspace(y_min, y_max, int(grid_size))
        for source_y in ys:
            for source_x in xs:
                columns = []
                for q_function in q_basis_functions:
                    unit = gaussian_puff_station_concentrations(
                        station_x_m=station_x_m,
                        station_y_m=station_y_m,
                        observation_times_h=observation_times,
                        wind_times_h=wind_times_h,
                        wind_u_mps=wind_u_mps,
                        wind_v_mps=wind_v_mps,
                        source_x_m=float(source_x),
                        source_y_m=float(source_y),
                        q_at_time=q_function,
                        diffusivity_m2s=diffusivity_m2s,
                        initial_sigma_m=initial_sigma_m,
                        decay_per_hour=decay_per_hour,
                        wind_factor=wind_factor,
                        pre_event_hours=pre_event_hours,
                        integration_dt_h=integration_dt_h,
                    )
                    columns.append(unit.reshape(-1))
                design = np.column_stack(columns)
                coefficients, rmse = profile_nonnegative_coefficients(
                    design, target.reshape(-1), iterations=nnls_iterations
                )
                evaluation_count += int(q_node_count)
                candidate = {
                    "source_x_m": float(source_x),
                    "source_y_m": float(source_y),
                    "q_node_values": [float(value) for value in coefficients],
                    "anomaly_rmse": rmse,
                    "level": level,
                }
                if return_candidates:
                    candidates.append(dict(candidate))
                if best is None or rmse < best["anomaly_rmse"]:
                    best = candidate
        if level + 1 < int(refinement_levels):
            dx = float(xs[1] - xs[0])
            dy = float(ys[1] - ys[0])
            x_min = max(x_domain_min, best["source_x_m"] - dx)
            x_max = min(x_domain_max, best["source_x_m"] + dx)
            y_min = max(y_domain_min, best["source_y_m"] - dy)
            y_max = min(y_domain_max, best["source_y_m"] + dy)
    target_rms = float(math.sqrt(np.mean(target**2)))
    result = {
        **best,
        "method": "B2_dynamic_Q_gaussian_puff_profile_grid",
        "q_parameterization": "nonnegative_piecewise_linear_nodes",
        "q_node_times_h": [float(value) for value in node_times],
        "normalized_anomaly_rmse": best["anomaly_rmse"] / max(target_rms, 1e-12),
        "grid_size": int(grid_size),
        "refinement_levels": int(refinement_levels),
        "forward_evaluation_count": evaluation_count,
        "source_bounds_m": [
            x_domain_min,
            x_domain_max,
            y_domain_min,
            y_domain_max,
        ],
        "physics": {
            "diffusivity_m2s": float(diffusivity_m2s),
            "initial_sigma_m": float(initial_sigma_m),
            "decay_per_hour": float(decay_per_hour),
            "wind_factor": float(wind_factor),
            "pre_event_hours": float(pre_event_hours),
            "integration_dt_h": float(integration_dt_h),
        },
    }
    if return_candidates:
        result["candidates"] = candidates
    return result
