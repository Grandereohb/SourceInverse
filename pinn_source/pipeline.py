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
    LOSS_W_RADIAL,
    LOSS_W_WIND,
    LOSS_W_BOUNDARY,
    LOSS_W_AXIS,
    AXIS_UPDATE_INTERVAL,
    D_MIN_PHYS,
    D_PERP_RATIO,
    RESIDUAL_R,
    RESIDUAL_W_SCALE,
    COLLOC_SOURCE_RATIO,
    COLLOC_SOURCE_R,
    WIND_SCALE,
    USE_ADAPTIVE_LOSS,
    ADAPTIVE_LOSS_LR,
    ADAPTIVE_INIT_LOG_VARS,
    ADAPTIVE_WARMUP_EPOCHS,
    ADAPTIVE_MIN_PRECISIONS,
    ADAPTIVE_MAX_PRECISIONS,
    DATA_NORMALIZE,
    DATA_SCALE_PERCENTILE,
    DATA_WARMUP_EPOCHS,
    DATA_WARMUP_PDE_FACTOR,
    PDE_RAMP_EPOCHS,
    MAX_GRAD_NORM,
)
from data_io import load_sites, load_wind, load_conc, wind_dir_to_uv
from model_registry import get_model
from adaptive_loss import AdaptiveLossWeights
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

    # Drop station columns that are all zero/NaN
    nonzero_station_cols = [
        c for c in station_cols if not data[c].replace(0, np.nan).isna().all()
    ]
    if not nonzero_station_cols:
        raise ValueError("All station columns are zero in the training data.")

    # Keep only needed columns
    data = data[["time", "dir", "sp"] + nonzero_station_cols]

    # Drop rows where wind dir is zero
    data = data.loc[data["dir"].fillna(0) != 0].copy()

    # Drop rows where any station concentration is zero/NaN
    mask_any_zero = data[nonzero_station_cols].replace(0, np.nan).isna().any(axis=1)
    data = data.loc[~mask_any_zero].copy()

    valid_stations = nonzero_station_cols
    sites_plot = sites[sites["station"].isin(valid_stations)].copy()
    if data.empty:
        raise ValueError("No matching timestamps between concentration and wind data.")

    # Build observation dataset
    obs = []
    for _, row in data.iterrows():
        u, v = wind_dir_to_uv(row["dir"], row["sp"], is_from=WIND_DIR_IS_FROM)
        for _, srow in sites_plot.iterrows():
            st = srow["station"]
            if st not in data.columns:
                continue
            obs.append([srow["x"], srow["y"], row["time"].timestamp(), row[st], u, v])
    if not obs:
        raise ValueError("No observation data matched station names. Check columns.")

    obs = np.array(obs, dtype=np.float64)
    x_obs = obs[:, 0]
    y_obs = obs[:, 1]
    t_obs = obs[:, 2]
    c_obs = obs[:, 3]
    u_obs = obs[:, 4]
    v_obs = obs[:, 5]

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
            n_terms=3,
            init_log_vars=ADAPTIVE_INIT_LOG_VARS,
            min_precisions=ADAPTIVE_MIN_PRECISIONS,
            max_precisions=ADAPTIVE_MAX_PRECISIONS,
        ).to(device)
        adaptive_opt = torch.optim.Adam(adaptive_loss.parameters(), lr=ADAPTIVE_LOSS_LR)

    x_obs_t = torch.tensor(x_obs, dtype=torch.float32, device=device).view(-1, 1)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=device).view(-1, 1)
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)

    # Robust concentration scaling for stable optimization
    c_scale = 1.0
    if DATA_NORMALIZE:
        c_scale = float(np.percentile(np.abs(c_obs), DATA_SCALE_PERCENTILE))
        if c_scale <= 0:
            c_scale = 1.0
    c_obs_scaled = c_obs / c_scale
    c_obs_t = torch.tensor(c_obs_scaled, dtype=torch.float32, device=device).view(-1, 1)
    c_obs_raw_t = torch.tensor(c_obs, dtype=torch.float32, device=device).view(-1, 1)

    print(
        f"Data summary: n_obs={len(c_obs)}, c_min={np.min(c_obs):.3f}, c_p50={np.percentile(c_obs, 50):.3f}, "
        f"c_p95={np.percentile(c_obs, 95):.3f}, c_max={np.max(c_obs):.3f}, c_scale={c_scale:.3f}"
    )

    # Precompute wind time series for collocation interpolation
    t_w = np.unique(t_obs)
    u_w = []
    v_w = []
    for tw in t_w:
        idx = np.where(t_obs == tw)[0][0]
        u_w.append(u_obs[idx])
        v_w.append(v_obs[idx])
    u_w = np.array(u_w)
    v_w = np.array(v_w)

    # Per-time wind vectors for time-sliced axis-loss on observation points
    t_w_t = torch.tensor(t_w, dtype=torch.float32, device=device).view(-1)
    u_w_t = torch.tensor(u_w, dtype=torch.float32, device=device).view(-1)
    v_w_t = torch.tensor(v_w, dtype=torch.float32, device=device).view(-1)

    rng = np.random.default_rng(0)

    def sample_collocation(xs_center, ys_center):
        n_src = int(N_COLLOCATION * COLLOC_SOURCE_RATIO)
        n_uni = N_COLLOCATION - n_src

        # Uniform background
        x_col_u = rng.uniform(x_min, x_max, n_uni)
        y_col_u = rng.uniform(y_min, y_max, n_uni)
        t_col_u = rng.uniform(t_min, t_max, n_uni)

        # Source-focused sampling around current estimate (xs, ys)
        x_col_s = rng.normal(loc=xs_center, scale=COLLOC_SOURCE_R, size=n_src)
        y_col_s = rng.normal(loc=ys_center, scale=COLLOC_SOURCE_R, size=n_src)
        x_col_s = np.clip(x_col_s, x_min, x_max)
        y_col_s = np.clip(y_col_s, y_min, y_max)
        t_col_s = rng.uniform(t_min, t_max, n_src)

        # Combine
        x_col = np.concatenate([x_col_u, x_col_s])
        y_col = np.concatenate([y_col_u, y_col_s])
        t_col = np.concatenate([t_col_u, t_col_s])

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

        # Data loss
        xyt_obs = torch.cat([x_obs_t, y_obs_t, t_obs_t], dim=1).requires_grad_(True)
        c_pred = model(xyt_obs)
        loss_data = torch.mean((c_pred - c_obs_t) ** 2)

        # PDE residual
        xyt_col = torch.cat([x_col_t, y_col_t, t_col_t], dim=1).requires_grad_(True)
        c_col = model(xyt_col)

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
        # D_norm_min = D_MIN_PHYS * T / (L**2)
        D = model.D() * T / (L**2)
        # D = torch.clamp(D, min=D_norm_min)

        # normalized source strength
        Q_col = model.Q(t_col_t) * T
        Q_mean = torch.mean(Q_col)

        # Advection term with time-varying wind (u,v)
        u_c = u_col_t
        v_c = v_col_t

        # Source term: approximate as a narrow Gaussian centered at (xs, ys)
        sigma_src = SIGMA_SRC
        xs = model.xs
        ys = model.ys
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
        # Weight residuals near source
        dx = xyt_col[:, 0:1] - xs
        dy = xyt_col[:, 1:2] - ys
        r = torch.sqrt(dx**2 + dy**2 + 1e-12)
        w = torch.exp(-(dx**2 + dy**2) / (2 * RESIDUAL_R**2))
        w = 1.0 + RESIDUAL_W_SCALE * w
        loss_pde = torch.mean((w * residual) ** 2)

        # Extra physically constrained source-identification losses
        # 1) Radial monotonicity: concentration should not increase outward from source.
        dc_dr = (dx * c_x + dy * c_y) / r
        loss_radial = torch.mean(torch.relu(dc_dr))

        # 3) Wind alignment: penalize concentration in upwind side of the source.
        projection = u_c * dx + v_c * dy
        upwind_mask = (projection < 0.0).squeeze(1)
        if torch.any(upwind_mask):
            loss_wind = torch.mean(torch.relu(c_col[upwind_mask]))
        else:
            loss_wind = torch.tensor(0.0, device=device)

        # 4) Plume-axis geometric constraint (time-sliced, observation points):
        # compute every N epochs and reuse cached scalar in between for speed.
        if (epoch - 1) % axis_update_every == 0:
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
        else:
            loss_axis = axis_loss_cache

        data_term = LOSS_W_DATA * loss_data
        pde_term = LOSS_W_PDE * loss_pde
        # penalty removed by request
        penalty_term = torch.tensor(0.0, device=device)
        radial_term = LOSS_W_RADIAL * loss_radial
        wind_term = LOSS_W_WIND * loss_wind
        axis_term = LOSS_W_AXIS * loss_axis

        # Boundary repulsion to avoid source collapsing to domain corners.
        dist_left = xs - x_min
        dist_right = x_max - xs
        dist_bottom = ys - y_min
        dist_top = y_max - ys
        min_dist_to_boundary = torch.min(
            torch.stack([dist_left, dist_right, dist_bottom, dist_top])
        ).clamp_min(1e-6)
        loss_boundary = 1.0 / min_dist_to_boundary
        boundary_term = LOSS_W_BOUNDARY * loss_boundary

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
            + penalty_term
            + radial_term
            + wind_term
            + boundary_term
            + axis_term
        )

        # Physical-unit RMSE for diagnostics
        # c_pred_phys = c_pred * c_scale
        # data_rmse = torch.sqrt(torch.mean((c_pred_phys - c_obs_raw_t) ** 2))
        # Important: only enable adaptive weighting after PDE warmup is over.
        # Otherwise pde_term_eff can be zero and adaptive refs become near-zero,
        # causing huge scaled losses once PDE term turns on.
        if adaptive_loss is not None and epoch > adaptive_start_epoch:
            train_loss, adaptive_weights = adaptive_loss(
                [data_term, pde_term_eff, penalty_term]
            )
            train_loss = (
                train_loss + radial_term + wind_term + boundary_term + axis_term
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

        if epoch % 500 == 0:
            extra = ""
            if adaptive_weights is not None:
                extra = (
                    f", w_data={adaptive_weights[0].item():.2f}"
                    f", w_pde={adaptive_weights[1].item():.2f}"
                    f", w_pen={adaptive_weights[2].item():.2f}"
                )
            elif adaptive_loss is not None:
                extra = ", adaptive=warmup"
            print(
                f"Epoch {epoch}: raw_loss={raw_loss.item():4f}, "
                f"data={loss_data.item():.4f}, pde={loss_pde.item():.4f}, "
                f"radial={loss_radial.item():.4f}, "
                f"wind={loss_wind.item():.4f}, boundary={loss_boundary.item():.4f}, axis={loss_axis.item():.4f}, "
                f"D={D.item():.3e}, Q_mean={Q_mean.item():.4f}, "
                f"xs={xs.item():.3f}, ys={ys.item():.3f}, pde_factor={pde_factor:.3f}"
                f"{extra}"
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
        c_scale=c_scale,
    )
