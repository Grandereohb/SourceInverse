"""DLQ author-style median wind aggregation, isolated from frozen baselines."""

import numpy as np
import pandas as pd


def fill_short_gaps(values, max_gap=5):
    """Interpolate complete interior gaps only, matching zoo maxgap semantics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional regular time series")
    result = values.copy()
    finite = np.isfinite(values)
    indices = np.flatnonzero(finite)
    for left, right in zip(indices[:-1], indices[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            result[left + 1:right] = np.linspace(values[left], values[right], gap + 2)[1:-1]
    return result


def aggregate_median_wind(speed, direction_deg):
    """Return ENU toward velocities; undefined directions remain NaN."""
    speed = np.array(speed, dtype=float, copy=True)
    direction_deg = np.asarray(direction_deg, dtype=float)
    if speed.shape != direction_deg.shape or speed.ndim != 2:
        raise ValueError("Expected matching time-by-station arrays")
    if np.any(speed[np.isfinite(speed)] < 0):
        raise ValueError("Wind speed cannot be negative")
    angle = np.deg2rad(direction_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    for column in range(speed.shape[1]):
        speed[:, column] = fill_short_gaps(speed[:, column])
        cosine[:, column] = fill_short_gaps(cosine[:, column])
        sine[:, column] = fill_short_gaps(sine[:, column])
    # DataFrame median handles all-missing rows without suppressing warnings.
    median = lambda x: pd.DataFrame(x).median(axis=1, skipna=True).to_numpy()
    magnitude = median(speed)
    x, y = median(cosine), median(sine)
    norm = np.hypot(x, y)
    valid = np.isfinite(magnitude) & np.isfinite(norm) & (norm > 1e-12)
    velocity = np.full((speed.shape[0], 2), np.nan)
    velocity[valid, 0] = -magnitude[valid] * y[valid] / norm[valid]
    velocity[valid, 1] = -magnitude[valid] * x[valid] / norm[valid]
    return velocity


def author_wind_table(raw, names=("E", "NW", "SW")):
    """Aggregate on the full minute clock before selecting any event window."""
    raw = raw[raw['name'].isin(names)].copy()
    raw['timestamp'] = pd.to_datetime(raw['time'], utc=True) - pd.Timedelta(minutes=1)
    # MAIN_1 explicitly keeps the first raw record at duplicated station-minutes.
    raw = raw.drop_duplicates(['timestamp', 'name'], keep='first')
    clock = pd.date_range(raw.timestamp.min(), raw.timestamp.max(), freq='min')
    def table(column):
        return raw.pivot(index='timestamp', columns='name', values=column).reindex(
            index=clock, columns=names).to_numpy(copy=True)
    velocity = aggregate_median_wind(table('wind.speed'), table('wind.direction'))
    return pd.DataFrame(velocity, index=clock, columns=['u', 'v'])
