import math
import time

import numpy as np
import pandas as pd
import torch

from field import predict_concentration
from geo_utils import xy_to_latlon


def _confidence_thresholds(prob_grid, levels):
    flat = prob_grid.reshape(-1)
    order = np.argsort(flat)[::-1]
    sorted_prob = flat[order]
    cumsum = np.cumsum(sorted_prob)
    thresholds = {}
    for level in levels:
        idx = np.searchsorted(cumsum, float(level), side="left")
        idx = min(max(idx, 0), len(sorted_prob) - 1)
        thresholds[str(level)] = float(sorted_prob[idx])
    return thresholds


def compute_source_loss_landscape(
    model,
    device,
    lon0,
    lat0,
    x0,
    y0,
    L,
    best_x_norm,
    best_y_norm,
    xyt_obs,
    u_obs_t,
    v_obs_t,
    c_obs_t,
    data_weight_t,
    sigma_src,
    radius_m,
    step_m,
    temperature,
    levels,
    x_bounds_m=None,
    y_bounds_m=None,
):

    best_x_m = best_x_norm * L + x0
    best_y_m = best_y_norm * L + y0
    if x_bounds_m is not None and y_bounds_m is not None:
        scan_mode = "source_domain"
        xs_m = np.arange(
            float(x_bounds_m[0]),
            float(x_bounds_m[1]) + 0.5 * float(step_m),
            float(step_m),
        )
        ys_m = np.arange(
            float(y_bounds_m[0]),
            float(y_bounds_m[1]) + 0.5 * float(step_m),
            float(step_m),
        )
    else:
        scan_mode = "local"
        offsets = np.arange(
            -float(radius_m), float(radius_m) + 0.5 * float(step_m), float(step_m)
        )
        xs_m = best_x_m + offsets
        ys_m = best_y_m + offsets

    orig_xs = model.xs.detach().clone()
    orig_ys = model.ys.detach().clone()
    rows = []
    total_candidates = int(len(xs_m) * len(ys_m))
    started_at = time.perf_counter()
    print(
        "Source landscape grid: "
        f"nx={len(xs_m)}, ny={len(ys_m)}, total={total_candidates}"
    )

    model.eval()
    with torch.no_grad():
        for iy, y_m in enumerate(ys_m):
            for x_m in xs_m:
                x_norm = (float(x_m) - x0) / L
                y_norm = (float(y_m) - y0) / L
                model.xs.data.fill_(x_norm)
                model.ys.data.fill_(y_norm)
                c_pred = predict_concentration(model, xyt_obs, u_obs_t, v_obs_t, sigma_src)
                data_loss = torch.mean(data_weight_t * ((c_pred - c_obs_t) ** 2))
                lon, lat = xy_to_latlon(float(x_m), float(y_m), lon0, lat0)
                rows.append(
                    {
                        "x": float(x_m),
                        "y": float(y_m),
                        "x_norm": float(x_norm),
                        "y_norm": float(y_norm),
                        "lon": float(lon),
                        "lat": float(lat),
                        "data_loss": float(data_loss.item()),
                        "loss": float(data_loss.item()),
                    }
                )
            if (iy + 1) % 10 == 0 or (iy + 1) == len(ys_m):
                done = (iy + 1) * len(xs_m)
                elapsed = time.perf_counter() - started_at
                rate = done / max(elapsed, 1e-9)
                remaining = (total_candidates - done) / max(rate, 1e-9)
                print(
                    "Source landscape progress: "
                    f"{done}/{total_candidates} "
                    f"({100.0 * done / total_candidates:.1f}%), "
                    f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s"
                )

        model.xs.data.copy_(orig_xs)
        model.ys.data.copy_(orig_ys)

    df = pd.DataFrame(rows)
    loss = df["loss"].to_numpy(dtype=float)
    delta = loss - float(np.min(loss))
    temp = max(float(temperature), 1e-8)
    prob = np.exp(-delta / temp)
    prob = prob / max(float(prob.sum()), 1e-12)
    df["probability"] = prob
    df["delta_loss"] = delta
    nx = len(xs_m)
    ny = len(ys_m)
    prob_grid = df["probability"].to_numpy(dtype=float).reshape(ny, nx)
    thresholds = _confidence_thresholds(prob_grid, levels)
    best_idx = int(df["loss"].idxmin())
    best_row = df.loc[best_idx].to_dict()
    trained_lon, trained_lat = xy_to_latlon(float(best_x_m), float(best_y_m), lon0, lat0)
    landscape_dx = float(best_row["x"]) - float(best_x_m)
    landscape_dy = float(best_row["y"]) - float(best_y_m)
    landscape_distance_m = float(math.sqrt(landscape_dx**2 + landscape_dy**2))
    best_boundary_margin_m = float(
        min(
            float(best_row["x"]) - float(xs_m[0]),
            float(xs_m[-1]) - float(best_row["x"]),
            float(best_row["y"]) - float(ys_m[0]),
            float(ys_m[-1]) - float(best_row["y"]),
        )
    )
    warnings = []
    if scan_mode == "source_domain":
        if landscape_distance_m > max(500.0, 2.0 * float(step_m)):
            warnings.append(
                "trained source and global loss-landscape best source are far apart"
            )
        if best_boundary_margin_m <= 1.5 * float(step_m):
            warnings.append(
                "global loss-landscape best source is close to the scan boundary; confidence contours may be truncated"
            )
    elif best_boundary_margin_m <= 1.5 * float(step_m):
        warnings.append(
            "local confidence best source is close to the local scan boundary; increase SOURCE_LANDSCAPE_RADIUS_M if needed"
        )

    if scan_mode == "local":
        interpretation = (
            "Probability contours describe local source uncertainty around the "
            "trained source with other learned parameters fixed."
        )
    else:
        interpretation = (
            "Probability contours describe the scanned global loss landscape around "
            "the global landscape best source, not necessarily uncertainty around "
            "the trained source."
        )

    payload = {
        "method": "single_run_loss_landscape",
        "scan_mode": scan_mode,
        "interpretation": interpretation,
        "radius_m": float(radius_m),
        "step_m": float(step_m),
        "temperature": float(temperature),
        "x_bounds_m": [float(xs_m[0]), float(xs_m[-1])],
        "y_bounds_m": [float(ys_m[0]), float(ys_m[-1])],
        "trained_source": {
            "x": float(best_x_m),
            "y": float(best_y_m),
            "lon": float(trained_lon),
            "lat": float(trained_lat),
        },
        "best": {
            "x": float(best_row["x"]),
            "y": float(best_row["y"]),
            "lon": float(best_row["lon"]),
            "lat": float(best_row["lat"]),
            "loss": float(best_row["loss"]),
        },
        "trained_to_landscape_best_distance_m": landscape_distance_m,
        "landscape_best_boundary_margin_m": best_boundary_margin_m,
        "warnings": warnings,
        "probability_thresholds": thresholds,
    }
    return payload, df
