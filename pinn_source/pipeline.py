import math
import numpy as np
import torch

from config import (
    EPOCHS,
    LR,
    N_COLLOCATION,
    DOMAIN_PAD_M,
    WIND_DIR_IS_FROM,
    MODEL_NAME,
    SIGMA_SRC,
    LOSS_W_DATA,
    LOSS_W_PDE,
    LOSS_W_BOUNDARY,
    ENABLE_LOSS_BOUNDARY,
    LOSS_W_AXIS,
    ENABLE_LOSS_AXIS,
    LOSS_W_SOURCE_LOCAL,
    ENABLE_LOSS_SOURCE_LOCAL,
    SOURCE_LOCAL_MARGIN,
    SOURCE_LOCAL_RING_R,
    AXIS_UPDATE_INTERVAL,
    PDE_SOURCE_MODE,
    D_MIN_PHYS,
    D_PERP_RATIO,
    RESIDUAL_R,
    RESIDUAL_W_SCALE,
    COLLOC_SOURCE_RATIO,
    COLLOC_PLUME_RATIO,
    COLLOC_SOURCE_R,
    COLLOC_PLUME_LENGTH,
    WIND_SCALE,
    USE_ADAPTIVE_LOSS,
    ADAPTIVE_LOSS_LR,
    ADAPTIVE_INIT_LOG_VARS,
    ADAPTIVE_WARMUP_EPOCHS,
    ADAPTIVE_MIN_PRECISIONS,
    ADAPTIVE_MAX_PRECISIONS,
    DATA_NORMALIZE,
    TRAIN_ON_RESIDUAL,
    BASELINE_MODE,
    DATA_SCALE_PERCENTILE,
    DATA_HIGH_WEIGHT,
    DATA_HIGH_PERCENTILE,
    DATA_HIGH_POWER,
    DATA_WARMUP_EPOCHS,
    DATA_WARMUP_PDE_FACTOR,
    PDE_RAMP_EPOCHS,
    MAX_GRAD_NORM,
    EARLY_STOP_START,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA,
    DEBUG_EVERY,
    LOSS_W_TOP_STATION,
    LOSS_W_HIGH_DOWNWIND,
    HIGH_DOWNWIND_RATIO,
    HIGH_DOWNWIND_MIN_RELIEF,
    HIGH_DOWNWIND_MARGIN,
)
from data_io import load_sites, load_wind, load_conc, wind_dir_to_uv
from model_registry import get_model
from adaptive_loss import AdaptiveLossWeights
from field import predict_concentration, field_components
from viz import plot_sites_and_source, diffusion_animation


def run(site_path, conc_path, wind_path):
    sites, lon0, lat0 = load_sites(site_path)
    wind = load_wind(wind_path)
    conc = load_conc(conc_path)

    # Merge on time
    data = conc.merge(wind, on="time", how="inner")

    # Only keep stations that actually have concentration data
    station_cols = [s for s in sites["station"].tolist() if s in data.columns]
    if not station_cols:
        raise ValueError("No station columns found in concentration data.")

    # Keep all station columns present in the input; zero is a valid value.
    valid_station_cols = [c for c in station_cols if not data[c].isna().all()]
    if not valid_station_cols:
        raise ValueError("All station columns are empty in the training data.")

    # Keep only needed columns
    data = data[["time", "dir", "sp"] + valid_station_cols]

    # Only drop rows with true missing values; zero is now treated as valid data.
    required_cols = ["dir", "sp"] + valid_station_cols
    data = data.dropna(subset=required_cols).copy()

    valid_stations = valid_station_cols
    sites_plot = sites[sites["station"].isin(valid_stations)].copy()
    if data.empty:
        raise ValueError("No matching timestamps between concentration and wind data.")

    station_matrix = data[valid_stations].astype(float)
    if TRAIN_ON_RESIDUAL:
        if BASELINE_MODE == "q25":
            baseline_series = station_matrix.quantile(0.25, axis=1)
        elif BASELINE_MODE == "q40":
            baseline_series = station_matrix.quantile(0.40, axis=1)
        else:
            baseline_series = station_matrix.median(axis=1)
    else:
        baseline_series = np.zeros(len(data), dtype=np.float64)
    baseline_vals = baseline_series.to_numpy(dtype=np.float64)

    # Build observation dataset
    obs = []
    for row_idx, (_, row) in enumerate(data.iterrows()):
        u, v = wind_dir_to_uv(row["dir"], row["sp"], is_from=WIND_DIR_IS_FROM)
        for _, srow in sites_plot.iterrows():
            st = srow["station"]
            if st not in data.columns:
                continue
            c_raw = float(row[st])
            c_fit = max(c_raw - baseline_vals[row_idx], 0.0) if TRAIN_ON_RESIDUAL else c_raw
            obs.append(
                [
                    srow["x"],
                    srow["y"],
                    row["time"].timestamp(),
                    c_fit,
                    c_raw,
                    baseline_vals[row_idx],
                    u,
                    v,
                ]
            )
    if not obs:
        raise ValueError("No observation data matched station names. Check columns.")

    obs = np.array(obs, dtype=np.float64)
    x_obs = obs[:, 0]
    y_obs = obs[:, 1]
    t_obs = obs[:, 2]
    c_obs = obs[:, 3]
    c_obs_raw = obs[:, 4]
    c_obs_baseline = obs[:, 5]
    u_obs = obs[:, 6]
    v_obs = obs[:, 7]

    # Normalize time to start at 0 and in hours
    t0 = np.min(t_obs)
    t_obs = (t_obs - t0) / 3600.0

    # Physical domain bounds (meters, hours)
    x_min_p, x_max_p = sites["x"].min(), sites["x"].max()
    y_min_p, y_max_p = sites["y"].min(), sites["y"].max()
    t_min_p, t_max_p = np.min(t_obs), np.max(t_obs)

    # Expand domain a bit (meters)
    pad = DOMAIN_PAD_M
    x_min_p -= pad
    x_max_p += pad
    y_min_p -= pad
    y_max_p += pad

    # Normalization scales
    L = x_max_p - x_min_p
    if L == 0:
        L = 1.0
    T = t_max_p - t_min_p
    if T == 0:
        T = 1.0

    x0 = 0.5 * (x_min_p + x_max_p)
    y0 = 0.5 * (y_min_p + y_max_p)
    t0_p = t_min_p

    # Normalize x, y, t to comparable scale
    x_obs = (x_obs - x0) / L
    y_obs = (y_obs - y0) / L
    t_obs = (t_obs - t0_p) / T

    # Scale wind to match normalized coordinates
    u_obs = u_obs * T / L * WIND_SCALE
    v_obs = v_obs * T / L * WIND_SCALE

    # Collocation points for PDE residual (normalized bounds)
    x_min, x_max = (x_min_p - x0) / L, (x_max_p - x0) / L
    y_min, y_max = (y_min_p - y0) / L, (y_max_p - y0) / L
    t_min, t_max = (t_min_p - t0_p) / T, (t_max_p - t0_p) / T

    # Build tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelCls = get_model(MODEL_NAME)
    model = ModelCls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    adaptive_loss = None
    adaptive_opt = None
    adaptive_start_epoch = max(
        ADAPTIVE_WARMUP_EPOCHS, DATA_WARMUP_EPOCHS + PDE_RAMP_EPOCHS
    )
    if USE_ADAPTIVE_LOSS:
        adaptive_loss = AdaptiveLossWeights(
            n_terms=2,
            init_log_vars=ADAPTIVE_INIT_LOG_VARS,
            min_precisions=ADAPTIVE_MIN_PRECISIONS,
            max_precisions=ADAPTIVE_MAX_PRECISIONS,
        ).to(device)
        adaptive_opt = torch.optim.Adam(adaptive_loss.parameters(), lr=ADAPTIVE_LOSS_LR)

    x_obs_t = torch.tensor(x_obs, dtype=torch.float32, device=device).view(-1, 1)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=device).view(-1, 1)
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)
    u_obs_t = torch.tensor(u_obs, dtype=torch.float32, device=device).view(-1, 1)
    v_obs_t = torch.tensor(v_obs, dtype=torch.float32, device=device).view(-1, 1)

    # Robust concentration scaling for stable optimization
    c_scale = 1.0
    if DATA_NORMALIZE:
        c_scale = float(np.percentile(np.abs(c_obs), DATA_SCALE_PERCENTILE))
        if c_scale <= 0:
            c_scale = 1.0
    c_obs_scaled = c_obs / c_scale
    c_obs_t = torch.tensor(c_obs_scaled, dtype=torch.float32, device=device).view(-1, 1)
    c_obs_raw_t = torch.tensor(c_obs_raw, dtype=torch.float32, device=device).view(-1, 1)
    c_obs_baseline_t = torch.tensor(c_obs_baseline, dtype=torch.float32, device=device).view(
        -1, 1
    )
    if DATA_HIGH_WEIGHT > 0:
        high_ref = float(np.percentile(c_obs, DATA_HIGH_PERCENTILE))
        high_ref = max(high_ref, 1e-6)
        high_excess = np.clip((c_obs - high_ref) / high_ref, a_min=0.0, a_max=None)
        data_weights = 1.0 + DATA_HIGH_WEIGHT * np.power(high_excess, DATA_HIGH_POWER)
    else:
        high_ref = 0.0
        data_weights = np.ones_like(c_obs)
    data_weight_t = torch.tensor(data_weights, dtype=torch.float32, device=device).view(-1, 1)
    print(
        f"Data summary: n_obs={len(c_obs)}, fit_min={np.min(c_obs):.3f}, fit_p50={np.percentile(c_obs, 50):.3f}, "
        f"fit_p95={np.percentile(c_obs, 95):.3f}, fit_max={np.max(c_obs):.3f}, c_scale={c_scale:.3f}, "
        f"train_on_residual={TRAIN_ON_RESIDUAL}, baseline_mode={BASELINE_MODE}"
    )
    if TRAIN_ON_RESIDUAL:
        print(
            f"Baseline summary: base_min={np.min(baseline_vals):.3f}, "
            f"base_p50={np.percentile(baseline_vals, 50):.3f}, "
            f"base_p95={np.percentile(baseline_vals, 95):.3f}, "
            f"base_max={np.max(baseline_vals):.3f}, "
            f"raw_max={np.max(c_obs_raw):.3f}"
        )
    print(
        f"Anomaly-weight summary: high_ref={high_ref:.3f}, "
        f"w_mean={np.mean(data_weights):.3f}, w_max={np.max(data_weights):.3f}"
    )

    # Precompute wind time series for collocation interpolation
    t_w = np.unique(t_obs)
    u_w = []
    v_w = []
    baseline_w = []
    for tw in t_w:
        idx = np.where(t_obs == tw)[0][0]
        u_w.append(u_obs[idx])
        v_w.append(v_obs[idx])
        baseline_w.append(c_obs_baseline[idx])
    u_w = np.array(u_w)
    v_w = np.array(v_w)
    baseline_w = np.array(baseline_w)

    # Per-time wind vectors for time-sliced axis-loss on observation points
    t_w_t = torch.tensor(t_w, dtype=torch.float32, device=device).view(-1)
    u_w_t = torch.tensor(u_w, dtype=torch.float32, device=device).view(-1)
    v_w_t = torch.tensor(v_w, dtype=torch.float32, device=device).view(-1)

    rng = np.random.default_rng(0)

    def sample_collocation(xs_center, ys_center):
        n_src = int(N_COLLOCATION * COLLOC_SOURCE_RATIO)
        n_plume = int(N_COLLOCATION * COLLOC_PLUME_RATIO)
        n_uni = max(0, N_COLLOCATION - n_src - n_plume)

        # 1) Global uniform coverage.
        x_col_u = rng.uniform(x_min, x_max, n_uni)
        y_col_u = rng.uniform(y_min, y_max, n_uni)
        t_col_u = rng.uniform(t_min, t_max, n_uni)

        # 2) Near-source polar sampling.
        t_col_s = rng.uniform(t_min, t_max, n_src)
        r_src = COLLOC_SOURCE_R * np.sqrt(rng.uniform(0.0, 1.0, n_src))
        theta_src = rng.uniform(0.0, 2.0 * math.pi, n_src)
        x_col_s = xs_center + r_src * np.cos(theta_src)
        y_col_s = ys_center + r_src * np.sin(theta_src)
        x_col_s = np.clip(x_col_s, x_min, x_max)
        y_col_s = np.clip(y_col_s, y_min, y_max)

        # 3) Downwind plume-axis sampling guided by time-varying wind.
        t_col_p = rng.uniform(t_min, t_max, n_plume)
        u_col_p = np.interp(t_col_p, t_w, u_w)
        v_col_p = np.interp(t_col_p, t_w, v_w)
        speed_p = np.sqrt(u_col_p**2 + v_col_p**2 + 1e-12)
        ex_p = u_col_p / speed_p
        ey_p = v_col_p / speed_p
        dist_p = rng.uniform(0.0, COLLOC_PLUME_LENGTH, n_plume)
        cross_p = rng.normal(loc=0.0, scale=0.15 * COLLOC_SOURCE_R, size=n_plume)
        nx_p = -ey_p
        ny_p = ex_p
        x_col_p = xs_center + dist_p * ex_p + cross_p * nx_p
        y_col_p = ys_center + dist_p * ey_p + cross_p * ny_p
        x_col_p = np.clip(x_col_p, x_min, x_max)
        y_col_p = np.clip(y_col_p, y_min, y_max)

        # Combine
        x_col = np.concatenate([x_col_u, x_col_s, x_col_p])
        y_col = np.concatenate([y_col_u, y_col_s, y_col_p])
        t_col = np.concatenate([t_col_u, t_col_s, t_col_p])

        # Build tensors
        x_col_t = torch.tensor(x_col, dtype=torch.float32, device=device).view(-1, 1)
        y_col_t = torch.tensor(y_col, dtype=torch.float32, device=device).view(-1, 1)
        t_col_t = torch.tensor(t_col, dtype=torch.float32, device=device).view(-1, 1)

        # Interpolate wind at collocation times
        u_col = np.interp(t_col, t_w, u_w)
        v_col = np.interp(t_col, t_w, v_w)
        u_col_t = torch.tensor(u_col, dtype=torch.float32, device=device).view(-1, 1)
        v_col_t = torch.tensor(v_col, dtype=torch.float32, device=device).view(-1, 1)

        return x_col_t, y_col_t, t_col_t, u_col_t, v_col_t

    # Initial collocation around domain center
    xs0 = 0.5 * (x_min + x_max)
    ys0 = 0.5 * (y_min + y_max)
    x_col_t, y_col_t, t_col_t, u_col_t, v_col_t = sample_collocation(xs0, ys0)

    axis_update_every = max(1, int(AXIS_UPDATE_INTERVAL))
    axis_loss_cache = torch.tensor(0.0, device=device)
    best_raw_loss = float("inf")
    early_stop_wait = 0

    def tensor_stats(name, tensor):
        flat = tensor.detach().view(-1)
        return (
            f"{name}[mean={flat.mean().item():.4f}, std={flat.std(unbiased=False).item():.4f}, "
            f"min={flat.min().item():.4f}, max={flat.max().item():.4f}]"
        )

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        if adaptive_opt is not None:
            adaptive_opt.zero_grad()

        # Dynamic collocation resampling around current source estimate
        if epoch % 200 == 1:
            xs_center = model.xs.detach().item()
            ys_center = model.ys.detach().item()
            x_col_t, y_col_t, t_col_t, u_col_t, v_col_t = sample_collocation(
                xs_center, ys_center
            )

        # Data loss uses the plain observation graph; expensive station curvature is handled separately.
        xyt_obs = torch.cat([x_obs_t, y_obs_t, t_obs_t], dim=1)
        xs = model.xs
        ys = model.ys
        c_pred = predict_concentration(model, xyt_obs, u_obs_t, v_obs_t, SIGMA_SRC)
        bg_obs, plume_obs, q_obs, gate_obs, source_obs = field_components(
            model, xyt_obs, u_obs_t, v_obs_t, SIGMA_SRC
        )
        data_residual = c_pred - c_obs_t
        loss_data = torch.mean(data_weight_t * (data_residual**2))
        if LOSS_W_TOP_STATION > 0:
            top_station_losses = []
            for i, t_i in enumerate(t_w_t):
                mask_t = torch.isclose(t_obs_t.view(-1), t_i, atol=1e-6, rtol=0.0)
                if not torch.any(mask_t):
                    continue
                pred_slice = c_pred.view(-1)[mask_t]
                obs_slice = c_obs_t.view(-1)[mask_t]
                top_idx = int(torch.argmax(obs_slice).item())
                top_pred = pred_slice[top_idx]
                if pred_slice.numel() > 1:
                    other_mask = torch.ones_like(pred_slice, dtype=torch.bool)
                    other_mask[top_idx] = False
                    other_max = torch.max(pred_slice[other_mask])
                    top_station_losses.append(torch.relu(other_max - top_pred))
            if top_station_losses:
                loss_top_station = torch.mean(torch.stack(top_station_losses))
            else:
                loss_top_station = torch.tensor(0.0, device=device)
        else:
            loss_top_station = torch.tensor(0.0, device=device)

        if LOSS_W_HIGH_DOWNWIND > 0:
            high_downwind_losses = []
            t_obs_flat = t_obs_t.view(-1)
            c_obs_flat = c_obs_t.view(-1)
            x_obs_flat = x_obs_t.view(-1)
            y_obs_flat = y_obs_t.view(-1)
            for i, t_i in enumerate(t_w_t):
                mask_t = torch.isclose(t_obs_flat, t_i, atol=1e-6, rtol=0.0)
                if not torch.any(mask_t):
                    continue
                obs_slice = c_obs_flat[mask_t]
                x_slice = x_obs_flat[mask_t]
                y_slice = y_obs_flat[mask_t]

                obs_max = torch.max(obs_slice).detach()
                obs_min = torch.min(obs_slice).detach()
                relief = (obs_max - obs_min) / torch.clamp(torch.abs(obs_max), min=1e-6)
                if float(relief.item()) < HIGH_DOWNWIND_MIN_RELIEF:
                    continue

                high_cut = HIGH_DOWNWIND_RATIO * obs_max
                high_mask = obs_slice >= high_cut
                if not torch.any(high_mask):
                    continue

                u_i = u_w_t[i]
                v_i = v_w_t[i]
                w_norm = torch.sqrt(u_i**2 + v_i**2 + 1e-12)
                w_x = u_i / w_norm
                w_y = v_i / w_norm

                dx_high = x_slice[high_mask] - xs
                dy_high = y_slice[high_mask] - ys
                dot_high = dx_high * w_x + dy_high * w_y
                high_weight = (
                    torch.relu(obs_slice[high_mask]) / torch.clamp(obs_max, min=1e-6)
                )
                high_downwind_losses.append(
                    torch.mean(high_weight * torch.relu(HIGH_DOWNWIND_MARGIN - dot_high))
                )

            if high_downwind_losses:
                loss_high_downwind = torch.mean(torch.stack(high_downwind_losses))
            else:
                loss_high_downwind = torch.tensor(0.0, device=device)
        else:
            loss_high_downwind = torch.tensor(0.0, device=device)

        # PDE residual
        xyt_col = torch.cat([x_col_t, y_col_t, t_col_t], dim=1).requires_grad_(True)
        c_col = predict_concentration(model, xyt_col, u_col_t, v_col_t, SIGMA_SRC)

        grads = torch.autograd.grad(
            c_col, xyt_col, torch.ones_like(c_col), create_graph=True
        )[0]
        c_x = grads[:, 0:1]
        c_y = grads[:, 1:2]
        c_t = grads[:, 2:3]

        c_xx = torch.autograd.grad(
            c_x, xyt_col, torch.ones_like(c_x), create_graph=True
        )[0][:, 0:1]
        c_xy = torch.autograd.grad(
            c_x, xyt_col, torch.ones_like(c_x), create_graph=True
        )[0][:, 1:2]
        c_yy = torch.autograd.grad(
            c_y, xyt_col, torch.ones_like(c_y), create_graph=True
        )[0][:, 1:2]

        # normalized diffusion coefficient
        # D_MIN_PHYS is in physical units, so convert its lower bound to normalized space.
        D_norm_min = D_MIN_PHYS * T / (L**2)
        # Keep D learnable above a physical floor instead of hard-clamping it flat.
        D = D_norm_min + model.D() * T / (L**2)

        # normalized source strength
        Q_col = model.Q(t_col_t) * T
        Q_mean = torch.mean(Q_col)

        # Advection term with time-varying wind (u,v)
        u_c = u_col_t
        v_c = v_col_t

        # Source term: approximate as a narrow Gaussian centered at (xs, ys)
        sigma_src = SIGMA_SRC
        if PDE_SOURCE_MODE == "gaussian":
            src = (
                Q_col
                / (2 * math.pi * sigma_src**2)
                * torch.exp(
                    -((xyt_col[:, 0:1] - xs) ** 2 + (xyt_col[:, 1:2] - ys) ** 2)
                    / (2 * sigma_src**2)
                )
            )
            # Keep PDE consistent when c is normalized
            src = src / c_scale
        else:
            src = torch.zeros_like(c_col)

        # Anisotropic diffusion aligned with wind direction:
        # learn D_parallel only, and set D_perp = D_PERP_RATIO * D_parallel.
        D_parallel = D
        D_perp = D_PERP_RATIO * D_parallel

        wind_speed = torch.sqrt(u_c**2 + v_c**2 + 1e-12)
        ex = u_c / wind_speed
        ey = v_c / wind_speed

        # Second derivative along wind direction s.
        c_ss = ex**2 * c_xx + 2.0 * ex * ey * c_xy + ey**2 * c_yy

        # Equivalent anisotropic diffusion term:
        # D_perp * Laplacian + (D_parallel - D_perp) * directional_curvature_along_wind
        lap = c_xx + c_yy
        diffusion_term = D_perp * lap + (D_parallel - D_perp) * c_ss

        residual = c_t + u_c * c_x + v_c * c_y - diffusion_term - src
        residual_abs_mean = torch.mean(torch.abs(residual))
        # Weight residuals near source
        dx = xyt_col[:, 0:1] - xs
        dy = xyt_col[:, 1:2] - ys
        w = torch.exp(-(dx**2 + dy**2) / (2 * RESIDUAL_R**2))
        w = 1.0 + RESIDUAL_W_SCALE * w
        loss_pde = torch.mean((w * residual) ** 2)

        # Extra physically constrained source-identification losses
        # 1) Source-local dominance: concentration at source center should exceed a nearby annulus.
        if ENABLE_LOSS_SOURCE_LOCAL and LOSS_W_SOURCE_LOCAL > 0:
            t_probe = t_w_t.view(-1, 1)
            center_x = xs.expand_as(t_probe)
            center_y = ys.expand_as(t_probe)
            center_pts = torch.cat([center_x, center_y, t_probe], dim=1)
            center_u = u_w_t.view(-1, 1)
            center_v = v_w_t.view(-1, 1)
            c_center = predict_concentration(
                model, center_pts, center_u, center_v, SIGMA_SRC
            )

            theta = torch.linspace(
                0.0, 2.0 * math.pi, steps=12, device=device, dtype=torch.float32
            )[:-1]
            ring_r = max(SOURCE_LOCAL_RING_R, SIGMA_SRC * 2.0)
            ring_x = xs + ring_r * torch.cos(theta)
            ring_y = ys + ring_r * torch.sin(theta)
            ring_pts = []
            for t_i in t_w_t:
                t_row = torch.full_like(theta, t_i)
                ring_pts.append(torch.stack([ring_x, ring_y, t_row], dim=1))
            ring_pts = torch.cat(ring_pts, dim=0)
            ring_u = center_u.repeat_interleave(theta.numel(), dim=0)
            ring_v = center_v.repeat_interleave(theta.numel(), dim=0)
            c_ring = predict_concentration(model, ring_pts, ring_u, ring_v, SIGMA_SRC)
            loss_source_local = torch.relu(
                torch.mean(c_ring) + SOURCE_LOCAL_MARGIN - torch.mean(c_center)
            )
        else:
            loss_source_local = torch.tensor(0.0, device=device)

        # 2) Plume-axis geometric constraint (time-sliced, observation points):
        # compute every N epochs and reuse cached scalar in between for speed.
        if ENABLE_LOSS_AXIS and LOSS_W_AXIS > 0 and (epoch - 1) % axis_update_every == 0:
            c_obs_flat = c_pred.view(-1)
            x_obs_flat = x_obs_t.view(-1)
            y_obs_flat = y_obs_t.view(-1)
            t_obs_flat = t_obs_t.view(-1)
            axis_losses = []
            axis_min_rel_contrast = 0.05

            for i in range(t_w_t.numel()):
                t_i = t_w_t[i]
                mask_t = torch.isclose(t_obs_flat, t_i, atol=1e-6, rtol=0.0)
                n_t = int(mask_t.sum().item())
                if n_t == 0:
                    continue

                c_slice = c_obs_flat[mask_t]
                x_slice = x_obs_flat[mask_t]
                y_slice = y_obs_flat[mask_t]

                # Skip axis loss when station concentrations are too similar.
                c_max = torch.max(c_slice).detach()
                c_min = torch.min(c_slice).detach()
                rel_contrast = (c_max - c_min) / torch.clamp(torch.abs(c_max), min=1e-6)
                if float(rel_contrast.item()) < axis_min_rel_contrast:
                    continue

                q90 = torch.quantile(c_slice.detach(), 0.9)
                high_mask = c_slice >= q90

                # If only one high-concentration point is selected, include the second-largest point.
                if int(high_mask.sum().item()) == 1 and n_t >= 2:
                    top2 = torch.topk(c_slice.detach(), k=2).indices
                    high_mask[top2[1]] = True

                w_c = torch.relu(c_slice[high_mask]) + 1e-8
                w_sum = torch.sum(w_c).clamp_min(1e-8)
                x_h = torch.sum(w_c * x_slice[high_mask]) / w_sum
                y_h = torch.sum(w_c * y_slice[high_mask]) / w_sum

                d_x = x_h - xs
                d_y = y_h - ys

                u_i = u_w_t[i]
                v_i = v_w_t[i]
                w_norm = torch.sqrt(u_i**2 + v_i**2 + 1e-12)
                w_x = u_i / w_norm
                w_y = v_i / w_norm

                dot = d_x * w_x + d_y * w_y
                margin = 0.03 * w_norm
                axis_losses.append(torch.relu(margin - dot))

            if len(axis_losses) > 0:
                loss_axis = torch.mean(torch.stack(axis_losses))
            else:
                loss_axis = torch.tensor(0.0, device=device)
            axis_loss_cache = loss_axis.detach()
        elif ENABLE_LOSS_AXIS and LOSS_W_AXIS > 0:
            loss_axis = axis_loss_cache
        else:
            loss_axis = torch.tensor(0.0, device=device)

        data_term = LOSS_W_DATA * loss_data
        pde_term = LOSS_W_PDE * loss_pde
        source_local_term = LOSS_W_SOURCE_LOCAL * loss_source_local
        axis_term = LOSS_W_AXIS * loss_axis
        top_station_term = LOSS_W_TOP_STATION * loss_top_station
        high_downwind_term = LOSS_W_HIGH_DOWNWIND * loss_high_downwind

        # Boundary repulsion to avoid source collapsing to domain corners.
        dist_left = xs - x_min
        dist_right = x_max - xs
        dist_bottom = ys - y_min
        dist_top = y_max - ys
        min_dist_to_boundary = torch.min(
            torch.stack([dist_left, dist_right, dist_bottom, dist_top])
        ).clamp_min(1e-6)
        loss_boundary = 1.0 / min_dist_to_boundary
        if ENABLE_LOSS_BOUNDARY and LOSS_W_BOUNDARY > 0:
            boundary_term = LOSS_W_BOUNDARY * loss_boundary
        else:
            boundary_term = torch.tensor(0.0, device=device)

        # Data-oriented warmup + smooth PDE ramp to avoid catastrophic forgetting
        if epoch <= DATA_WARMUP_EPOCHS:
            pde_factor = DATA_WARMUP_PDE_FACTOR
        else:
            ramp = min(1.0, (epoch - DATA_WARMUP_EPOCHS) / max(1, PDE_RAMP_EPOCHS))
            pde_factor = DATA_WARMUP_PDE_FACTOR + (1.0 - DATA_WARMUP_PDE_FACTOR) * ramp
        pde_term_eff = pde_factor * pde_term

        raw_loss = (
            data_term
            + pde_term_eff
            + source_local_term
            + boundary_term
            + axis_term
            + top_station_term
            + high_downwind_term
        )

        # Physical-unit RMSE for diagnostics
        # c_pred_phys = c_pred * c_scale
        # data_rmse = torch.sqrt(torch.mean((c_pred_phys - c_obs_raw_t) ** 2))
        # Important: only enable adaptive weighting after PDE warmup is over.
        # Otherwise pde_term_eff can be zero and adaptive refs become near-zero,
        # causing huge scaled losses once PDE term turns on.
        if adaptive_loss is not None and epoch > adaptive_start_epoch:
            train_loss, adaptive_weights = adaptive_loss([data_term, pde_term_eff])
            train_loss = (
                train_loss
                + source_local_term
                + boundary_term
                + axis_term
                + top_station_term
                + high_downwind_term
            )
        else:
            train_loss = raw_loss
            adaptive_weights = None

        train_loss.backward()

        # Clip gradients to reduce sudden divergence after PDE ramps up
        if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # Track source-position gradient magnitudes for diagnostics
        # grad_xs = 0.0
        # grad_ys = 0.0
        # if model.xs.grad is not None:
        #     grad_xs = float(model.xs.grad.detach().abs().item())
        # if model.ys.grad is not None:
        #     grad_ys = float(model.ys.grad.detach().abs().item())

        opt.step()
        if adaptive_opt is not None and epoch > adaptive_start_epoch:
            adaptive_opt.step()

        raw_loss_value = float(raw_loss.detach().item())
        if epoch >= EARLY_STOP_START:
            if raw_loss_value < best_raw_loss - EARLY_STOP_MIN_DELTA:
                best_raw_loss = raw_loss_value
                early_stop_wait = 0
            else:
                early_stop_wait += 1
                if early_stop_wait >= EARLY_STOP_PATIENCE:
                    print(
                        f"Early stop at epoch {epoch}: raw_loss stabilized at {best_raw_loss:.6f}"
                    )
                    break

        if epoch % 500 == 0:
            extra = ""
            if adaptive_weights is not None:
                extra = (
                    f", w_data={adaptive_weights[0].item():.2f}"
                    f", w_pde={adaptive_weights[1].item():.2f}"
                )
            elif adaptive_loss is not None:
                extra = ", adaptive=warmup"
            loss_parts = [
                f"data={loss_data.item():.4f}",
                f"pde={loss_pde.item():.4f}",
            ]
            if ENABLE_LOSS_SOURCE_LOCAL and LOSS_W_SOURCE_LOCAL > 0:
                loss_parts.append(f"source_local={loss_source_local.item():.4f}")
            if ENABLE_LOSS_BOUNDARY and LOSS_W_BOUNDARY > 0:
                loss_parts.append(f"boundary={loss_boundary.item():.4f}")
            if ENABLE_LOSS_AXIS and LOSS_W_AXIS > 0:
                loss_parts.append(f"axis={loss_axis.item():.4f}")
            if LOSS_W_TOP_STATION > 0:
                loss_parts.append(f"top_station={loss_top_station.item():.4f}")
            if LOSS_W_HIGH_DOWNWIND > 0:
                loss_parts.append(f"high_downwind={loss_high_downwind.item():.4f}")
            print(
                f"Epoch {epoch}: raw_loss={raw_loss.item():4f}, "
                f"{', '.join(loss_parts)}, "
                f"D={D.item():.3e}, Q_mean={Q_mean.item():.4f}, "
                f"xs={xs.item():.3f}, ys={ys.item():.3f}, pde_factor={pde_factor:.3f}"
                f"{extra}"
            )
        if DEBUG_EVERY > 0 and epoch % DEBUG_EVERY == 0:
            if TRAIN_ON_RESIDUAL:
                pred_raw = c_pred * c_scale + c_obs_baseline_t
            else:
                pred_raw = c_pred * c_scale
            print(
                "Debug: "
                + ", ".join(
                    [
                        tensor_stats("bg", bg_obs),
                        tensor_stats("plume", plume_obs),
                        tensor_stats("Q", q_obs),
                        tensor_stats("gate", gate_obs),
                        tensor_stats("source_term", source_obs),
                        tensor_stats("pred", c_pred),
                        f"fit_raw_rmse={torch.sqrt(torch.mean((pred_raw - c_obs_raw_t) ** 2)).item():.4f}",
                        f"weighted_data_residual={torch.mean(torch.sqrt(data_weight_t) * torch.abs(data_residual)).item():.4f}",
                        f"bg_abs_mean={torch.mean(torch.abs(bg_obs)).item():.4f}",
                        f"source_abs_mean={torch.mean(torch.abs(source_obs)).item():.4f}",
                        f"pde_abs_mean={residual_abs_mean.item():.4f}",
                        f"D={D.item():.4e}",
                        f"D_parallel={D_parallel.mean().item():.4e}",
                        f"D_perp={D_perp.mean().item():.4e}",
                    ]
                )
            )

    # Convert predicted source back to lat/lon
    xs = model.xs.item()
    ys = model.ys.item()
    xs_p = xs * L + x0
    ys_p = ys * L + y0
    pred_lon = lon0 + xs_p / (math.cos(math.radians(lat0)) * 111320.0)
    pred_lat = lat0 + ys_p / 110540.0

    print("Estimated source (x,y) meters:", xs_p, ys_p)
    print("Estimated source (lat,lon):", pred_lat, pred_lon)

    plot_sites_and_source(sites_plot, pred_lon, pred_lat)

    diffusion_animation(
        model,
        device,
        x_min_p,
        x_max_p,
        y_min_p,
        y_max_p,
        t_min_p,
        t_max_p,
        lon0,
        lat0,
        sites_plot,
        pred_lon,
        pred_lat,
        x0=x0,
        y0=y0,
        t0=t0_p,
        L=L,
        T=T,
        t_w=t_w,
        u_w=u_w,
        v_w=v_w,
        baseline_w=baseline_w,
        sigma_src=SIGMA_SRC,
        c_scale=c_scale,
    )
