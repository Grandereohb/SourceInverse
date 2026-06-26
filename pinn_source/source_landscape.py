import json
import math
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from field import predict_concentration
from geo_utils import xy_to_latlon

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
mpl.rcParams["axes.unicode_minus"] = False


def _source_geometry_score(
    xs,
    ys,
    obs_time_indices,
    x_obs_flat,
    y_obs_flat,
    c_obs_flat,
    u_w_t,
    v_w_t,
    axis_high_ratio,
    axis_min_relief,
    axis_along_margin,
    axis_cross_base,
    axis_cross_slope,
    high_downwind_ratio,
    high_downwind_min_relief,
    high_downwind_margin,
):
    scores = []
    for i, idx_t in enumerate(obs_time_indices):
        if idx_t.numel() == 0:
            continue

        obs_slice = c_obs_flat[idx_t]
        x_slice = x_obs_flat[idx_t]
        y_slice = y_obs_flat[idx_t]
        obs_max = torch.max(obs_slice)
        obs_min = torch.min(obs_slice)
        relief = (obs_max - obs_min) / torch.clamp(torch.abs(obs_max), min=1e-6)

        u_i = u_w_t[i]
        v_i = v_w_t[i]
        w_norm = torch.sqrt(u_i**2 + v_i**2 + 1e-12)
        w_x = u_i / w_norm
        w_y = v_i / w_norm

        if float(relief.item()) >= high_downwind_min_relief:
            high_mask = obs_slice >= (high_downwind_ratio * obs_max)
            if torch.any(high_mask):
                dx_high = x_slice[high_mask] - xs
                dy_high = y_slice[high_mask] - ys
                dot_high = dx_high * w_x + dy_high * w_y
                high_weight = torch.relu(obs_slice[high_mask]) / torch.clamp(obs_max, min=1e-6)
                scores.append(torch.mean(high_weight * torch.relu(high_downwind_margin - dot_high)))

        if float(relief.item()) >= axis_min_relief:
            high_mask = obs_slice >= (axis_high_ratio * obs_max)
            if torch.any(high_mask):
                dx_high = x_slice[high_mask] - xs
                dy_high = y_slice[high_mask] - ys
                along_high = dx_high * w_x + dy_high * w_y
                cross_high = torch.abs(-dx_high * w_y + dy_high * w_x)
                high_weight = torch.relu(obs_slice[high_mask]) / torch.clamp(obs_max, min=1e-6)
                corridor_half_width = axis_cross_base + axis_cross_slope * torch.relu(along_high)
                forward_penalty = torch.relu(axis_along_margin - along_high)
                cross_penalty = torch.relu(cross_high - corridor_half_width)
                scores.append(torch.mean(high_weight * (forward_penalty + cross_penalty)))

    if not scores:
        return xs * 0.0
    return torch.mean(torch.stack(scores))


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


def _median_grid_spacing(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(np.unique(values)))
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _needs_display_kernel(prob_grid):
    prob_grid = np.asarray(prob_grid, dtype=float)
    finite = prob_grid[np.isfinite(prob_grid)]
    if finite.size == 0:
        return False
    max_prob = float(np.max(finite))
    if max_prob <= 0.0:
        return False
    active_cells = int(np.sum(finite >= max_prob * 0.05))
    return max_prob >= 0.75 or active_cells <= 3


def _smooth_probability_surface(lon_grid, lat_grid, prob_grid, upscale=45, sigma=1.4):
    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)
    prob_grid = np.asarray(prob_grid, dtype=float)
    lon_centers = np.nanmean(lon_grid, axis=0)
    lat_centers = np.nanmean(lat_grid, axis=1)

    lon_order = np.argsort(lon_centers)
    lat_order = np.argsort(lat_centers)
    lon_centers = lon_centers[lon_order]
    lat_centers = lat_centers[lat_order]
    prob_grid = prob_grid[np.ix_(lat_order, lon_order)]

    n_lon = max(int(len(lon_centers) * upscale), 80)
    n_lat = max(int(len(lat_centers) * upscale), 80)
    fine_lon = np.linspace(float(lon_centers[0]), float(lon_centers[-1]), n_lon)
    fine_lat = np.linspace(float(lat_centers[0]), float(lat_centers[-1]), n_lat)
    fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)

    if _needs_display_kernel(prob_grid):
        iy, ix = np.unravel_index(np.nanargmax(prob_grid), prob_grid.shape)
        center_lon = float(lon_centers[ix])
        center_lat = float(lat_centers[iy])
        sigma_lon = max(_median_grid_spacing(lon_centers) * 0.6, 1e-12)
        sigma_lat = max(_median_grid_spacing(lat_centers) * 0.6, 1e-12)
        smooth = np.exp(
            -0.5
            * (
                ((fine_lon_grid - center_lon) / sigma_lon) ** 2
                + ((fine_lat_grid - center_lat) / sigma_lat) ** 2
            )
        )
        smooth = smooth / float(np.sum(smooth))
        return fine_lon_grid, fine_lat_grid, smooth

    interp_lon = np.vstack(
        [np.interp(fine_lon, lon_centers, row) for row in prob_grid]
    )
    smooth = np.vstack(
        [np.interp(fine_lat, lat_centers, interp_lon[:, i]) for i in range(n_lon)]
    ).T

    sigma = max(float(sigma), 0.0)
    if sigma > 0.0:
        radius = max(1, int(3.0 * sigma))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-(x**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        smooth = np.apply_along_axis(
            lambda arr: np.convolve(arr, kernel, mode="same"), axis=0, arr=smooth
        )
        smooth = np.apply_along_axis(
            lambda arr: np.convolve(arr, kernel, mode="same"), axis=1, arr=smooth
        )

    smooth = np.clip(smooth, 0.0, None)
    total = float(np.sum(smooth))
    if total > 0.0:
        smooth = smooth / total
    return fine_lon_grid, fine_lat_grid, smooth


def _draw_probability_regions(
    ax,
    lon_grid,
    lat_grid,
    prob_grid,
    thresholds,
    levels,
    colors="#00e5ff",
    linewidths=2.4,
    fontsize=8,
    zorder=3,
):
    if not thresholds or not levels:
        return

    fine_lon, fine_lat, smooth_prob = _smooth_probability_surface(
        lon_grid, lat_grid, prob_grid
    )
    smooth_thresholds = _confidence_thresholds(smooth_prob, levels)
    contour_pairs = []

    for level in sorted([float(v) for v in levels]):
        key = str(level)
        if key not in smooth_thresholds:
            continue

        threshold = float(smooth_thresholds[key])
        if not np.isfinite(threshold):
            continue
        contour_pairs.append((threshold, f"{int(level * 100)}%"))

    unique_pairs = []
    for threshold, label in sorted(contour_pairs, key=lambda item: item[0]):
        if unique_pairs and np.isclose(threshold, unique_pairs[-1][0]):
            unique_pairs[-1] = (
                unique_pairs[-1][0],
                f"{unique_pairs[-1][1]}/{label}",
            )
        else:
            unique_pairs.append((threshold, label))

    if not unique_pairs:
        return

    cs = ax.contour(
        fine_lon,
        fine_lat,
        smooth_prob,
        levels=[p[0] for p in unique_pairs],
        colors=colors,
        linewidths=linewidths,
        zorder=zorder + 0.5,
    )
    ax.clabel(
        cs,
        inline=True,
        fontsize=fontsize,
        fmt={threshold: label for threshold, label in unique_pairs},
    )


def compute_source_loss_landscape(
    model,
    device,
    output_dir,
    sites_plot,
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
    obs_time_indices,
    t_w_t,
    u_w_t,
    v_w_t,
    x_obs_t,
    y_obs_t,
    sigma_src,
    radius_m,
    step_m,
    temperature,
    levels,
    include_geometry,
    geometry_kwargs,
    x_bounds_m=None,
    y_bounds_m=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        x_obs_flat = x_obs_t.view(-1)
        y_obs_flat = y_obs_t.view(-1)
        c_obs_flat = c_obs_t.view(-1)

        for iy, y_m in enumerate(ys_m):
            for x_m in xs_m:
                x_norm = (float(x_m) - x0) / L
                y_norm = (float(y_m) - y0) / L
                model.xs.data.fill_(x_norm)
                model.ys.data.fill_(y_norm)
                c_pred = predict_concentration(model, xyt_obs, u_obs_t, v_obs_t, sigma_src)
                data_loss = torch.mean(data_weight_t * ((c_pred - c_obs_t) ** 2))
                geom_loss = torch.tensor(0.0, device=device)
                if include_geometry:
                    geom_loss = _source_geometry_score(
                        model.xs,
                        model.ys,
                        obs_time_indices,
                        x_obs_flat,
                        y_obs_flat,
                        c_obs_flat,
                        u_w_t,
                        v_w_t,
                        **geometry_kwargs,
                    )
                total = data_loss + geom_loss
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
                        "geometry_loss": float(geom_loss.item()),
                        "loss": float(total.item()),
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
    df.to_csv(output_dir / "source_loss_landscape.csv", index=False, encoding="utf-8-sig")
    df.to_csv(output_dir / "source_probability_map.csv", index=False, encoding="utf-8-sig")

    nx = len(xs_m)
    ny = len(ys_m)
    prob_grid = df["probability"].to_numpy(dtype=float).reshape(ny, nx)
    loss_grid = df["loss"].to_numpy(dtype=float).reshape(ny, nx)
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
    with open(output_dir / "source_confidence_landscape.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _save_landscape_plot(
        output_dir=output_dir,
        sites_plot=sites_plot,
        xs_m=xs_m,
        ys_m=ys_m,
        prob_grid=prob_grid,
        loss_grid=loss_grid,
        thresholds=thresholds,
        levels=levels,
        lon0=lon0,
        lat0=lat0,
        best_row=best_row,
        trained_source=payload["trained_source"],
        warnings=warnings,
        scan_mode=scan_mode,
    )
    return payload


def _save_landscape_plot(
    output_dir,
    sites_plot,
    xs_m,
    ys_m,
    prob_grid,
    loss_grid,
    thresholds,
    levels,
    lon0,
    lat0,
    best_row,
    trained_source,
    warnings,
    scan_mode,
):
    lon_grid = np.zeros((len(ys_m), len(xs_m)))
    lat_grid = np.zeros((len(ys_m), len(xs_m)))
    for iy, y in enumerate(ys_m):
        for ix, x in enumerate(xs_m):
            lon, lat = xy_to_latlon(float(x), float(y), lon0, lat0)
            lon_grid[iy, ix] = lon
            lat_grid[iy, ix] = lat

    fig, ax = plt.subplots(figsize=(8, 7))
    fine_lon, fine_lat, smooth_prob = _smooth_probability_surface(
        lon_grid, lat_grid, prob_grid
    )
    smooth_display = smooth_prob / max(float(np.nanmax(smooth_prob)), 1e-12)
    display_prob = np.ma.masked_less(
        smooth_display, 0.02
    )
    mesh = ax.contourf(
        fine_lon,
        fine_lat,
        display_prob,
        levels=18,
        cmap="YlOrRd",
        alpha=0.34,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Relative source probability")

    _draw_probability_regions(
        ax,
        lon_grid,
        lat_grid,
        prob_grid,
        thresholds,
        levels,
        colors="#00e5ff",
        linewidths=2.4,
        fontsize=8,
        zorder=3,
    )

    ax.contour(lon_grid, lat_grid, loss_grid, levels=8, colors="black", alpha=0.25, linewidths=0.8)
    ax.scatter(sites_plot["lon"], sites_plot["lat"], c="#2563eb", s=55, label="Stations", zorder=4)
    for _, row in sites_plot.iterrows():
        ax.text(row["lon"], row["lat"], str(row["station"]), fontsize=8, ha="left", va="bottom")

    ax.scatter(
        [best_row["lon"]],
        [best_row["lat"]],
        marker="X",
        c="#f97316",
        s=180,
        edgecolors="black",
        linewidths=0.6,
        label=(
            "Local landscape best"
            if scan_mode == "local"
            else "Best landscape source"
        ),
        zorder=5,
    )
    ax.scatter(
        [trained_source["lon"]],
        [trained_source["lat"]],
        marker="*",
        c="red",
        s=180,
        edgecolors="black",
        linewidths=0.6,
        label="Trained source",
        zorder=6,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = (
        "Local Source Confidence Landscape"
        if scan_mode == "local"
        else "Fast Source Loss Landscape"
    )
    if warnings:
        title += " (warning)"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "source_confidence_landscape.png", dpi=220)
    plt.close(fig)
