"""Deterministic-seed Lagrangian particle receptor model for stress tests."""

from __future__ import annotations

import math

import numpy as np


SECONDS_PER_HOUR = 3600.0


def lagrangian_particle_station_concentrations(
    *,
    station_x_m,
    station_y_m,
    observation_times_h,
    wind_times_h,
    wind_u_mps,
    wind_v_mps,
    source_x_m,
    source_y_m,
    q_at_time,
    diffusivity_along_m2s,
    diffusivity_cross_m2s,
    initial_sigma_along_m,
    initial_sigma_cross_m,
    sensor_kernel_sigma_m=60.0,
    wind_factor=1.0,
    decay_per_hour=0.0,
    meander_amplitude_m=0.0,
    meander_period_h=6.0,
    pre_event_hours=1.0,
    release_dt_h=0.1,
    particles_per_release=64,
    random_seed=0,
):
    """Evaluate an anisotropic, optionally meandering particle plume at receptors.

    The ensemble uses fixed antithetic normal draws. Reusing each draw across ages
    gives deterministic, smooth marginal particle clouds; this is a benchmark
    stress generator, not a replacement for a meteorology-driven LPDM.
    """
    station_x = np.asarray(station_x_m, dtype=np.float64).reshape(-1)
    station_y = np.asarray(station_y_m, dtype=np.float64).reshape(-1)
    observation_times = np.asarray(observation_times_h, dtype=np.float64).reshape(-1)
    wind_times = np.asarray(wind_times_h, dtype=np.float64).reshape(-1)
    wind_u = np.asarray(wind_u_mps, dtype=np.float64).reshape(-1)
    wind_v = np.asarray(wind_v_mps, dtype=np.float64).reshape(-1)
    if station_x.size == 0 or station_x.size != station_y.size:
        raise ValueError("station coordinates must be non-empty and paired")
    if observation_times.size == 0 or wind_times.size == 0:
        raise ValueError("observation and wind times must be non-empty")
    if wind_times.size != wind_u.size or wind_times.size != wind_v.size:
        raise ValueError("wind time and vector arrays must have equal length")
    if np.any(np.diff(observation_times) < 0.0) or np.any(np.diff(wind_times) < 0.0):
        raise ValueError("time arrays must be sorted")
    particle_count = int(particles_per_release)
    if particle_count < 4:
        raise ValueError("particles_per_release must be at least 4")
    half = (particle_count + 1) // 2
    rng = np.random.default_rng(int(random_seed))

    release_start = float(observation_times[0]) - max(float(pre_event_hours), 0.0)
    release_end = float(observation_times[-1])
    dt = max(float(release_dt_h), 1e-4)
    step_count = max(1, int(math.ceil((release_end - release_start) / dt)))
    release_times = np.linspace(release_start, release_end, step_count + 1)
    fine_dt = float(release_times[1] - release_times[0])
    effective_u = np.interp(release_times, wind_times, wind_u) * float(wind_factor)
    effective_v = np.interp(release_times, wind_times, wind_v) * float(wind_factor)
    cumulative_x = np.zeros_like(release_times)
    cumulative_y = np.zeros_like(release_times)
    cumulative_x[1:] = np.cumsum(
        0.5 * (effective_u[:-1] + effective_u[1:]) * SECONDS_PER_HOUR * fine_dt
    )
    cumulative_y[1:] = np.cumsum(
        0.5 * (effective_v[:-1] + effective_v[1:]) * SECONDS_PER_HOUR * fine_dt
    )
    q_release = np.asarray(q_at_time(release_times), dtype=np.float64).reshape(-1)
    if q_release.size != release_times.size:
        raise ValueError("q_at_time must return one value per requested time")
    if np.any(q_release < 0.0) or not np.all(np.isfinite(q_release)):
        raise ValueError("source strength must be finite and non-negative")

    draws = rng.standard_normal((release_times.size, half, 2))
    draws = np.concatenate([draws, -draws], axis=1)[:, :particle_count, :]
    sensor_sigma = max(float(sensor_kernel_sigma_m), 1e-6)
    sensor_variance = sensor_sigma**2
    d_along = max(float(diffusivity_along_m2s), 0.0)
    d_cross = max(float(diffusivity_cross_m2s), 0.0)
    sigma_along0 = max(float(initial_sigma_along_m), 1e-6)
    sigma_cross0 = max(float(initial_sigma_cross_m), 1e-6)
    decay = max(float(decay_per_hour), 0.0)
    meander_period = max(float(meander_period_h), 1e-6)
    concentrations = np.zeros((observation_times.size, station_x.size), dtype=np.float64)

    for obs_index, obs_time in enumerate(observation_times):
        mask = release_times <= obs_time + 1e-12
        cohort_times = release_times[mask]
        ages_h = np.maximum(obs_time - cohort_times, 0.0)
        obs_cumulative_x = np.interp(obs_time, release_times, cumulative_x)
        obs_cumulative_y = np.interp(obs_time, release_times, cumulative_y)
        displacement_x = obs_cumulative_x - cumulative_x[mask]
        displacement_y = obs_cumulative_y - cumulative_y[mask]
        distance = np.hypot(displacement_x, displacement_y)
        fallback_angle = np.arctan2(
            np.interp(obs_time, wind_times, wind_v),
            np.interp(obs_time, wind_times, wind_u),
        )
        along_x = np.where(distance > 1e-9, displacement_x / np.maximum(distance, 1e-9), np.cos(fallback_angle))
        along_y = np.where(distance > 1e-9, displacement_y / np.maximum(distance, 1e-9), np.sin(fallback_angle))
        cross_x = -along_y
        cross_y = along_x
        ramp = 1.0 - np.exp(-ages_h / 0.5)
        meander = float(meander_amplitude_m) * ramp * np.sin(
            2.0 * np.pi * (cohort_times - release_start) / meander_period
            + np.pi * ages_h / meander_period
        )
        center_x = float(source_x_m) + displacement_x + meander * cross_x
        center_y = float(source_y_m) + displacement_y + meander * cross_y
        age_seconds = ages_h * SECONDS_PER_HOUR
        std_along = np.sqrt(sigma_along0**2 + 2.0 * d_along * age_seconds)
        std_cross = np.sqrt(sigma_cross0**2 + 2.0 * d_cross * age_seconds)
        z = draws[mask]
        particle_x = center_x[:, None] + std_along[:, None] * z[:, :, 0] * along_x[:, None] + std_cross[:, None] * z[:, :, 1] * cross_x[:, None]
        particle_y = center_y[:, None] + std_along[:, None] * z[:, :, 0] * along_y[:, None] + std_cross[:, None] * z[:, :, 1] * cross_y[:, None]
        dx = station_x[None, None, :] - particle_x[:, :, None]
        dy = station_y[None, None, :] - particle_y[:, :, None]
        kernel = np.exp(-(dx**2 + dy**2) / (2.0 * sensor_variance))
        kernel /= 2.0 * np.pi * sensor_variance
        cohort_weights = q_release[mask] * np.exp(-decay * ages_h) * fine_dt
        concentrations[obs_index] = np.sum(
            cohort_weights[:, None, None] * kernel, axis=(0, 1)
        ) / particle_count
    return concentrations
