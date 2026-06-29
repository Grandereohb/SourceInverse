import numpy as np
import torch
import matplotlib as mpl
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt

from geo_utils import xy_to_latlon
from config import VISUALIZE_GATE_ONLY, ADD_BASELINE_TO_VIZ
from field import predict_concentration, field_components

# Font config for CJK labels
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False


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


def _probability_thresholds(prob_grid, levels):
    flat = np.asarray(prob_grid, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or float(np.sum(flat)) <= 0.0:
        return {}
    flat = np.sort(flat)[::-1]
    cumsum = np.cumsum(flat)
    thresholds = {}
    for level in levels:
        idx = np.searchsorted(cumsum, float(level), side="left")
        idx = min(max(idx, 0), len(flat) - 1)
        thresholds[str(level)] = float(flat[idx])
    return thresholds


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
    zorder=2,
):
    if not thresholds or not levels:
        return

    fine_lon, fine_lat, smooth_prob = _smooth_probability_surface(
        lon_grid, lat_grid, prob_grid
    )
    smooth_thresholds = _probability_thresholds(smooth_prob, levels)
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


def plot_sites_and_source(
    sites_plot,
    pred_lon,
    pred_lat,
    confidence_map=None,
    confidence_thresholds=None,
    confidence_levels=None,
    landscape_best=None,
    confidence_warnings=None,
    save_path=None,
    show=True,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    if confidence_map is not None:
        lon_grid = np.asarray(confidence_map.get("lon_grid"), dtype=float)
        lat_grid = np.asarray(confidence_map.get("lat_grid"), dtype=float)
        prob_grid = np.asarray(confidence_map.get("prob_grid"), dtype=float)
        if lon_grid.size and lat_grid.size and prob_grid.size:
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
                zorder=1,
            )
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Relative source probability")

            _draw_probability_regions(
                ax,
                lon_grid,
                lat_grid,
                prob_grid,
                confidence_thresholds,
                confidence_levels,
                colors="#00e5ff",
                linewidths=2.4,
                fontsize=8,
                zorder=2,
            )

    ax.scatter(
        sites_plot["lon"],
        sites_plot["lat"],
        c="blue",
        s=80,
        label="Stations",
        zorder=4,
    )
    for _, r in sites_plot.iterrows():
        ax.text(
            r["lon"], r["lat"], str(r["station"]), fontsize=10, ha="left", va="bottom"
        )
    ax.scatter(
        pred_lon,
        pred_lat,
        c="red",
        s=150,
        marker="*",
        label="Trained source",
        zorder=5,
    )
    if landscape_best is not None:
        ax.scatter(
            landscape_best["lon"],
            landscape_best["lat"],
            c="#f97316",
            s=120,
            marker="X",
            edgecolors="black",
            linewidths=0.6,
            label="Landscape best",
            zorder=6,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = (
        "Stations, Trained Source, and Global Loss Region"
        if confidence_map is not None
        else "Stations and Trained Source"
    )
    if confidence_warnings:
        title += " (warning)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def diffusion_animation(
    model,
    device,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    lon0,
    lat0,
    sites_plot,
    pred_lon,
    pred_lat,
    x0,
    y0,
    t0,
    L,
    T,
    t_w,
    u_w,
    v_w,
    baseline_w,
    sigma_src,
    c_scale=1.0,
    n_frames=40,
    nx=120,
    ny=120,
    out_gif="diffusion.gif",
    show=True,
):
    model.eval()
    xs_lin = np.linspace(x_min, x_max, nx)
    ys_lin = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs_lin, ys_lin)

    # Use time range in hours
    t_frames = np.linspace(t_min, t_max, n_frames)

    # Precompute frames so we can choose a good global color scale
    frames = []
    for tf in t_frames:
        tt = np.full_like(XX, tf)
        xyt = np.stack([XX.ravel(), YY.ravel(), tt.ravel()], axis=1)
        xyt[:, 0] = (xyt[:, 0] - x0) / L
        xyt[:, 1] = (xyt[:, 1] - y0) / L
        xyt[:, 2] = (xyt[:, 2] - t0) / T
        xyt_t = torch.tensor(xyt, dtype=torch.float32, device=device)
        u_tf = np.interp(tf, t_w, u_w)
        v_tf = np.interp(tf, t_w, v_w)
        u_t = torch.full((xyt_t.shape[0], 1), float(u_tf), dtype=torch.float32, device=device)
        v_t = torch.full((xyt_t.shape[0], 1), float(v_tf), dtype=torch.float32, device=device)
        with torch.no_grad():
            if VISUALIZE_GATE_ONLY:
                _, _, _, gate, _ = field_components(
                    model, xyt_t, u_t, v_t, sigma_src=sigma_src
                )
                cc = gate.cpu().numpy().reshape(ny, nx)
            else:
                cc = predict_concentration(
                    model, xyt_t, u_t, v_t, sigma_src=sigma_src
                ).cpu().numpy().reshape(ny, nx)
        cc = cc * c_scale
        if ADD_BASELINE_TO_VIZ and baseline_w is not None:
            cc = cc + float(np.interp(tf, t_w, baseline_w))
        cc = np.clip(cc, 0, None)  # for visualization
        frames.append(cc)

    # robust color scale
    all_vals = np.concatenate([f.ravel() for f in frames])
    vmin = np.percentile(all_vals, 5)
    vmax = np.percentile(all_vals, 95)
    if vmin == vmax:
        vmin, vmax = all_vals.min(), all_vals.max()

    lon_min, lat_min = xy_to_latlon(x_min, y_min, lon0, lat0)
    lon_max, lat_max = xy_to_latlon(x_max, y_max, lon0, lat0)

    # Ensure source is inside the view with a small padding
    lon_min = min(lon_min, pred_lon)
    lon_max = max(lon_max, pred_lon)
    lat_min = min(lat_min, pred_lat)
    lat_max = max(lat_max, pred_lat)
    pad_lon = (lon_max - lon_min) * 0.05
    pad_lat = (lat_max - lat_min) * 0.05
    lon_min -= pad_lon
    lon_max += pad_lon
    lat_min -= pad_lat
    lat_max += pad_lat

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        frames[0],
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="viridis",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(
        sites_plot["lon"],
        sites_plot["lat"],
        c="white",
        s=20,
        edgecolors="black",
        label="Stations",
    )
    for _, r in sites_plot.iterrows():
        ax.text(
            r["lon"],
            r["lat"],
            str(r["station"]),
            fontsize=9,
            color="white",
            ha="left",
            va="bottom",
        )
    ax.scatter(pred_lon, pred_lat, c="red", s=80, marker="*", label="Estimated Source")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Gate" if VISUALIZE_GATE_ONLY else "Concentration")

    def frame_fn(i):
        im.set_data(frames[i])
        ax.set_title(f"Diffusion Over Time (t={t_frames[i]:.2f} h)")
        return [im]

    ani = animation.FuncAnimation(
        fig, frame_fn, frames=n_frames, interval=200, blit=False
    )
    ani.save(out_gif, writer="pillow", fps=5)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_station_timeseries(
    times,
    station_names,
    obs_values,
    pred_values,
    title="Observed vs Predicted Concentration",
    save_path=None,
    show=True,
):
    times = np.asarray(times)
    station_names = np.asarray(station_names)
    obs_values = np.asarray(obs_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    unique_stations = list(dict.fromkeys(station_names.tolist()))
    n_station = len(unique_stations)
    if n_station == 0:
        return

    ncols = 2 if n_station > 1 else 1
    nrows = int(np.ceil(n_station / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 3.6 * nrows),
        squeeze=False,
        sharex=False,
    )
    axes_flat = axes.ravel()

    for ax, station in zip(axes_flat, unique_stations):
        mask = station_names == station
        station_times = pd.to_datetime(times[mask]).to_numpy()
        order = np.argsort(station_times)
        station_times = station_times[order]
        station_obs = obs_values[mask][order]
        station_pred = pred_values[mask][order]

        ax.plot(station_times, station_obs, color="#1f77b4", lw=1.8, label="Observed")
        ax.plot(station_times, station_pred, color="#d62728", lw=1.6, ls="--", label="Predicted")
        ax.set_title(str(station))
        ax.set_ylabel("Concentration")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25)

    for ax in axes_flat[n_station:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
