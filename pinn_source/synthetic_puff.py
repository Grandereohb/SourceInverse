"""Independent Gaussian-puff truth model for source-inversion benchmarks.

The implementation works directly at station locations and uses a fine release
time grid. It does not call the recurrent semi-Lagrangian inversion solver.
"""

from __future__ import annotations

import numpy as np


SECONDS_PER_HOUR = 3600.0


def gaussian_puff_station_concentrations(
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
    diffusivity_m2s,
    initial_sigma_m,
    decay_per_hour=0.0,
    wind_factor=1.0,
    pre_event_hours=1.0,
    integration_dt_h=0.025,
):
    """Integrate isotropic Gaussian puffs released by a stationary point source."""
    station_x = np.asarray(station_x_m, dtype=np.float64).reshape(-1)
    station_y = np.asarray(station_y_m, dtype=np.float64).reshape(-1)
    observation_times = np.asarray(observation_times_h, dtype=np.float64).reshape(-1)
    wind_times = np.asarray(wind_times_h, dtype=np.float64).reshape(-1)
    wind_u = np.asarray(wind_u_mps, dtype=np.float64).reshape(-1)
    wind_v = np.asarray(wind_v_mps, dtype=np.float64).reshape(-1)
    if station_x.size != station_y.size:
        raise ValueError("station x/y arrays must have equal length")
    if wind_times.size != wind_u.size or wind_times.size != wind_v.size:
        raise ValueError("wind time/u/v arrays must have equal length")
    if observation_times.size == 0 or wind_times.size == 0:
        raise ValueError("observation and wind time arrays must be non-empty")
    if np.any(np.diff(observation_times) < 0.0) or np.any(np.diff(wind_times) < 0.0):
        raise ValueError("time arrays must be sorted")

    dt = max(float(integration_dt_h), 1e-4)
    release_start = float(observation_times[0]) - max(float(pre_event_hours), 0.0)
    release_end = float(observation_times[-1])
    steps = max(1, int(np.ceil((release_end - release_start) / dt)))
    release_times = np.linspace(release_start, release_end, steps + 1)
    fine_dt = float(release_times[1] - release_times[0])
    effective_u = np.interp(release_times, wind_times, wind_u) * float(wind_factor)
    effective_v = np.interp(release_times, wind_times, wind_v) * float(wind_factor)

    cumulative_x = np.zeros_like(release_times)
    cumulative_y = np.zeros_like(release_times)
    cumulative_x[1:] = np.cumsum(
        0.5 * (effective_u[:-1] + effective_u[1:])
        * SECONDS_PER_HOUR
        * fine_dt
    )
    cumulative_y[1:] = np.cumsum(
        0.5 * (effective_v[:-1] + effective_v[1:])
        * SECONDS_PER_HOUR
        * fine_dt
    )
    q_release = np.asarray(q_at_time(release_times), dtype=np.float64).reshape(-1)
    if q_release.size != release_times.size:
        raise ValueError("q_at_time must return one value per requested time")
    if np.any(q_release < 0.0) or not np.all(np.isfinite(q_release)):
        raise ValueError("source strength must be finite and non-negative")

    sigma0_sq = max(float(initial_sigma_m), 1e-6) ** 2
    diffusion = max(float(diffusivity_m2s), 0.0)
    decay = max(float(decay_per_hour), 0.0)
    concentrations = np.zeros(
        (observation_times.size, station_x.size), dtype=np.float64
    )
    for obs_index, obs_time in enumerate(observation_times):
        mask = release_times <= obs_time + 1e-12
        release = release_times[mask]
        ages_h = np.maximum(obs_time - release, 0.0)
        obs_cumulative_x = np.interp(obs_time, release_times, cumulative_x)
        obs_cumulative_y = np.interp(obs_time, release_times, cumulative_y)
        center_x = float(source_x_m) + obs_cumulative_x - cumulative_x[mask]
        center_y = float(source_y_m) + obs_cumulative_y - cumulative_y[mask]
        variance = sigma0_sq + 2.0 * diffusion * SECONDS_PER_HOUR * ages_h
        dx = station_x[None, :] - center_x[:, None]
        dy = station_y[None, :] - center_y[:, None]
        kernel = np.exp(-(dx**2 + dy**2) / (2.0 * variance[:, None]))
        kernel /= 2.0 * np.pi * variance[:, None]
        kernel *= np.exp(-decay * ages_h)[:, None]
        weights = np.ones(release.size, dtype=np.float64)
        if release.size > 1:
            weights[[0, -1]] = 0.5
        concentrations[obs_index] = np.sum(
            q_release[mask, None] * kernel * weights[:, None], axis=0
        ) * fine_dt
    return concentrations


def source_strength_shape(
    times_h,
    kind="single_pulse",
    reference_start_h=None,
    reference_duration_h=None,
):
    """Return a dimensionless, non-negative benchmark source-strength shape."""
    times = np.asarray(times_h, dtype=np.float64)
    if times.size == 0:
        return times.copy()
    start = (
        float(np.min(times))
        if reference_start_h is None
        else float(reference_start_h)
    )
    span = (
        max(float(np.max(times) - start), 1.0)
        if reference_duration_h is None
        else max(float(reference_duration_h), 1e-12)
    )
    phase = (times - start) / span
    if kind == "constant":
        values = np.ones_like(phase)
    elif kind == "step":
        values = np.where(phase < 0.45, 0.25, 1.0)
    elif kind == "double_peak":
        values = 0.15 + np.exp(-0.5 * ((phase - 0.30) / 0.09) ** 2)
        values += 0.75 * np.exp(-0.5 * ((phase - 0.72) / 0.11) ** 2)
    elif kind == "single_pulse":
        values = 0.12 + np.exp(-0.5 * ((phase - 0.45) / 0.14) ** 2)
    else:
        raise ValueError(f"Unknown source-strength shape: {kind}")
    return np.maximum(values, 0.0)
