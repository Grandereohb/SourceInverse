"""Common-background preprocessing and its nonnegative-prediction error floor."""

import numpy as np
import pandas as pd


def estimate_common_background(station_matrix, enabled=True, mode="median"):
    """Keep the existing cross-station estimator; disabled means raw targets."""
    if mode not in ("median", "q25", "q40"):
        raise ValueError("Baseline mode must be median, q25 or q40.")
    if not enabled:
        return pd.Series(0.0, index=station_matrix.index, dtype=np.float64)
    if mode == "median":
        return station_matrix.median(axis=1)
    return station_matrix.quantile(.25 if mode == "q25" else .40, axis=1)


def nonnegative_background_floor(observations, background, stations, predictions):
    """For prediction=b+p, p>=0, SSE >= sum(max(b-y,0)^2). Diagnostic only."""
    y, b, pred = (np.asarray(v, dtype=np.float64).reshape(-1)
                  for v in (observations, background, predictions))
    labels = np.asarray(stations, dtype=str).reshape(-1)
    if (not y.size or any(v.shape != y.shape for v in (b, pred, labels))
            or not np.isfinite(np.r_[y, b, pred]).all()):
        raise ValueError("Finite, equally sized observations/background/predictions required.")
    floor = np.maximum(b-y, 0.)
    sse = float(np.dot(floor, floor))
    error_sse = float(np.dot(pred-y, pred-y))
    tolerance = 1e-6 * max(1., float(np.abs(np.r_[b, pred]).max()))
    applicable = bool(np.all(pred >= b-tolerance))
    return {
        "raw_rmse_lower_bound": float(np.sqrt(sse / y.size)),
        "raw_sse_lower_bound": sse,
        "fraction_of_current_raw_sse": sse/error_sse if applicable and error_sse > 0 else None,
        "nonnegative_plume_assumption_satisfied": applicable,
        "observation_count": int(y.size),
        "observations_below_background": int(np.sum(y < b)),
        "station_raw_rmse_lower_bounds": {
            label: float(np.sqrt(np.mean(floor[labels == label]**2)))
            for label in np.unique(labels)
        },
        "note": "Algebraic floor for the selected background and nonnegative plume; not a physical background validation or a training penalty.",
    }
