import math
import os
import time
import numpy as np
import torch

from config import (
    EPOCHS,
    LR,
    N_COLLOCATION,
    DOMAIN_PAD_M,
    WIND_DIR_IS_FROM,
    MODEL_NAME,
    DEVICE,
    SIGMA_SRC,
    LOSS_W_DATA,
    LOSS_W_PDE,
    LOSS_W_AXIS,
    ENABLE_LOSS_AXIS,
    AXIS_MIN_RELIEF,
    AXIS_HIGH_RATIO,
    AXIS_ALONG_MARGIN,
    AXIS_CROSS_BASE,
    AXIS_CROSS_SLOPE,
    LOSS_W_SOURCE_LOCAL,
    ENABLE_LOSS_SOURCE_LOCAL,
    SOURCE_LOCAL_MARGIN,
    SOURCE_LOCAL_RING_R,
    AXIS_UPDATE_INTERVAL,
    AUX_LOSS_UPDATE_INTERVAL,
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
    ENABLE_EVENT_WINDOW_CROP,
    EVENT_WINDOW_MIN_MAX,
    EVENT_WINDOW_MIN_RELIEF,
    EVENT_WINDOW_PAD_STEPS,
    DATA_SCALE_PERCENTILE,
    DATA_HIGH_WEIGHT,
    DATA_HIGH_PERCENTILE,
    DATA_HIGH_POWER,
    DATA_TIME_PEAK_WEIGHT,
    DATA_TIME_PEAK_RATIO,
    DATA_TIME_PEAK_POWER,
    DATA_TIME_PEAK_MIN_RELIEF,
    EVENT_TIME_WEIGHT,
    EVENT_PEAK_WEIGHT,
    EVENT_PEAK_RATIO,
    DATA_WARMUP_EPOCHS,
    DATA_WARMUP_PDE_FACTOR,
    PDE_RAMP_EPOCHS,
    STAGE1_EPOCHS,
    STAGE1_PDE_FACTOR,
    STAGE1_DATA_MULT,
    STAGE1_TOP_STATION_MULT,
    STAGE1_MULTI_HIGH_MULT,
    STAGE1_HIGH_DOWNWIND_MULT,
    STAGE1_SOURCE_LOCAL_MULT,
    MAX_GRAD_NORM,
    EARLY_STOP_START,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA,
    DEBUG_EVERY,
    LOSS_W_TOP_STATION,
    LOSS_W_MULTI_HIGH,
    MULTI_HIGH_RATIO,
    MULTI_HIGH_MIN_RELIEF,
    MULTI_HIGH_MARGIN,
    LOSS_W_HIGH_DOWNWIND,
    HIGH_DOWNWIND_RATIO,
    HIGH_DOWNWIND_MIN_RELIEF,
    HIGH_DOWNWIND_MARGIN,
)
from data_io import load_sites, load_wind, load_conc, wind_dir_to_uv
from model_registry import get_model
from adaptive_loss import AdaptiveLossWeights
from field import predict_concentration, field_components
from viz import plot_sites_and_source, diffusion_animation, plot_station_timeseries


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

    # Keep only the main anomaly window if requested, with a small time padding on both ends.
    if ENABLE_EVENT_WINDOW_CROP:
        residual_matrix = np.clip(
            station_matrix.to_numpy(dtype=np.float64) - baseline_vals[:, None],
            a_min=0.0,
            a_max=None,
        )
        ts_max = residual_matrix.max(axis=1)
        ts_med = np.median(residual_matrix, axis=1)
        ts_relief = (ts_max - ts_med) / np.maximum(np.abs(ts_max), 1e-6)
        event_mask = (ts_max >= EVENT_WINDOW_MIN_MAX) & (
            ts_relief >= EVENT_WINDOW_MIN_RELIEF
        )

        event_indices = np.flatnonzero(event_mask)
        if event_indices.size > 0:
            start_idx = max(0, int(event_indices[0]) - int(EVENT_WINDOW_PAD_STEPS))
            end_idx = min(
                len(data) - 1,
                int(event_indices[-1]) + int(EVENT_WINDOW_PAD_STEPS),
            )
            keep_mask = np.zeros(len(data), dtype=bool)
            keep_mask[start_idx : end_idx + 1] = True
            data = data.loc[keep_mask].reset_index(drop=True)
            station_matrix = data[valid_stations].astype(float)
            baseline_series = baseline_series.loc[keep_mask].reset_index(drop=True)
            baseline_vals = baseline_series.to_numpy(dtype=np.float64)
            print(
                "Event window crop: "
                f"kept_rows={keep_mask.sum()}/{len(keep_mask)}, "
                f"start={data.loc[0, 'time']}, end={data.loc[len(data)-1, 'time']}, "
                f"pad_steps={EVENT_WINDOW_PAD_STEPS}"
            )
        else:
            print(
                "Event window crop: no anomaly window detected, using full time range."
            )

    # Build observation dataset
    obs = []
    obs_station_labels = []
    obs_time_labels = []
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
            obs_station_labels.append(st)
            obs_time_labels.append(row["time"])
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
    t_obs_group = obs[:, 2].copy()

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
    device_pref = str(os.environ.get("PINN_DEVICE", DEVICE)).strip().lower()
    if device_pref == "cpu":
        device = torch.device("cpu")
    elif device_pref == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (preference={device_pref})")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    def sync_device():
        if device.type == "cuda":
            torch.cuda.synchronize()

    ModelCls = get_model(MODEL_NAME)
    model = ModelCls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    adaptive_loss = None
    adaptive_opt = None
    adaptive_start_epoch = max(
        ADAPTIVE_WARMUP_EPOCHS, STAGE1_EPOCHS + PDE_RAMP_EPOCHS
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

    if DATA_TIME_PEAK_WEIGHT > 0:
        time_peak_weights = np.ones_like(c_obs)
        unique_obs_times = np.unique(t_obs_group)
        for t_key in unique_obs_times:
            mask_t = np.isclose(t_obs_group, t_key)
            c_slice = c_obs[mask_t]
            if c_slice.size == 0:
                continue
            c_max = float(np.max(c_slice))
            c_med = float(np.median(c_slice))
            relief = (c_max - c_med) / max(abs(c_max), 1e-6)
            if relief < DATA_TIME_PEAK_MIN_RELIEF:
                continue
            high_cut = DATA_TIME_PEAK_RATIO * c_max
            denom = max(c_max - high_cut, 1e-6)
            local_excess = np.clip((c_slice - high_cut) / denom, a_min=0.0, a_max=None)
            time_peak_weights[mask_t] = 1.0 + DATA_TIME_PEAK_WEIGHT * np.power(
                local_excess, DATA_TIME_PEAK_POWER
            )
        data_weights = data_weights * time_peak_weights
    else:
        time_peak_weights = np.ones_like(c_obs)

    event_time_weights = np.ones_like(c_obs)
    event_peak_weights = np.ones_like(c_obs)
    if EVENT_TIME_WEIGHT > 0 or EVENT_PEAK_WEIGHT > 0:
        unique_obs_times = np.unique(t_obs_group)
        for t_key in unique_obs_times:
            mask_t = np.isclose(t_obs_group, t_key)
            c_slice = c_obs[mask_t]
            if c_slice.size == 0:
                continue
            c_max = float(np.max(c_slice))
            c_med = float(np.median(c_slice))
            relief = (c_max - c_med) / max(abs(c_max), 1e-6)
            is_event_time = (c_max >= EVENT_WINDOW_MIN_MAX) and (
                relief >= EVENT_WINDOW_MIN_RELIEF
            )
            if not is_event_time:
                continue
            if EVENT_TIME_WEIGHT > 0:
                event_time_weights[mask_t] = EVENT_TIME_WEIGHT
            if EVENT_PEAK_WEIGHT > 0:
                peak_mask = c_slice >= (EVENT_PEAK_RATIO * c_max)
                idx_t = np.flatnonzero(mask_t)
                event_peak_weights[idx_t[peak_mask]] = EVENT_PEAK_WEIGHT
        data_weights = data_weights * event_time_weights * event_peak_weights
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
    if DATA_TIME_PEAK_WEIGHT > 0:
        print(
            f"Time-peak-weight summary: w_mean={np.mean(time_peak_weights):.3f}, "
            f"w_max={np.max(time_peak_weights):.3f}, ratio={DATA_TIME_PEAK_RATIO:.2f}, "
            f"min_relief={DATA_TIME_PEAK_MIN_RELIEF:.2f}"
        )
    if EVENT_TIME_WEIGHT > 0 or EVENT_PEAK_WEIGHT > 0:
        print(
            f"Event-weight summary: time_w_mean={np.mean(event_time_weights):.3f}, "
            f"time_w_max={np.max(event_time_weights):.3f}, "
            f"peak_w_mean={np.mean(event_peak_weights):.3f}, "
            f"peak_w_max={np.max(event_peak_weights):.3f}"
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
    obs_station_labels_arr = np.array(obs_station_labels, dtype=object)
    obs_time_indices = [
        torch.tensor(np.where(np.isclose(t_obs, tw))[0], dtype=torch.long, device=device)
        for tw in t_w
    ]
    station_time_indices = []
    for st in valid_stations:
        idx_s = np.flatnonzero(obs_station_labels_arr == st)
        if idx_s.size == 0:
            continue
        idx_s = idx_s[np.argsort(t_obs[idx_s])]
        station_time_indices.append(
            torch.tensor(idx_s, dtype=torch.long, device=device)
        )

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
    aux_loss_update_every = max(1, int(AUX_LOSS_UPDATE_INTERVAL))
    axis_loss_cache = torch.tensor(0.0, device=device)
    top_station_loss_cache = torch.tensor(0.0, device=device)
    multi_high_loss_cache = torch.tensor(0.0, device=device)
    high_downwind_loss_cache = torch.tensor(0.0, device=device)
    source_local_loss_cache = torch.tensor(0.0, device=device)
    best_raw_loss = float("inf")
    early_stop_wait = 0

    def tensor_stats(name, tensor):
        flat = tensor.detach().view(-1)
        return (
            f"{name}[mean={flat.mean().item():.4f}, std={flat.std(unbiased=False).item():.4f}, "
            f"min={flat.min().item():.4f}, max={flat.max().item():.4f}]"
        )

    def safe_scalar(value):
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    for epoch in range(1, EPOCHS + 1):
        sync_device()
        epoch_start_time = time.perf_counter()
        timing_data_forward = 0.0
        timing_obs_losses = 0.0
        timing_pde = 0.0
        timing_source_local = 0.0
        timing_axis = 0.0
        timing_backward = 0.0
        timing_optimizer = 0.0
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
        sync_device()
        t_section = time.perf_counter()
        xyt_obs = torch.cat([x_obs_t, y_obs_t, t_obs_t], dim=1)
        xs = model.xs
        ys = model.ys
        c_pred = predict_concentration(model, xyt_obs, u_obs_t, v_obs_t, SIGMA_SRC)
        bg_obs, plume_obs, q_obs, gate_obs, source_obs = field_components(
            model, xyt_obs, u_obs_t, v_obs_t, SIGMA_SRC
        )
        c_pred_flat = c_pred.view(-1)
        c_obs_flat = c_obs_t.view(-1)
        x_obs_flat = x_obs_t.view(-1)
        y_obs_flat = y_obs_t.view(-1)
        sync_device()
        timing_data_forward += time.perf_counter() - t_section

        sync_device()
        t_section = time.perf_counter()
        data_residual = c_pred - c_obs_t
        loss_data = torch.mean(data_weight_t * (data_residual**2))
        aux_should_update = (epoch - 1) % aux_loss_update_every == 0
        multi_high_shape_loss_dbg = None
        multi_high_sep_loss_dbg = None
        multi_high_time_loss_dbg = None
        if LOSS_W_TOP_STATION > 0 and aux_should_update:
            top_station_losses = []
            for idx_t in obs_time_indices:
                if idx_t.numel() == 0:
                    continue
                pred_slice = c_pred_flat[idx_t]
                obs_slice = c_obs_flat[idx_t]
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
            top_station_loss_cache = loss_top_station.detach()
        elif LOSS_W_TOP_STATION > 0:
            loss_top_station = top_station_loss_cache
        else:
            loss_top_station = torch.tensor(0.0, device=device)

        if LOSS_W_MULTI_HIGH > 0 and aux_should_update:
            multi_high_losses = []
            multi_high_shape_terms = []
            multi_high_sep_terms = []
            for idx_t in obs_time_indices:
                if idx_t.numel() == 0:
                    continue
                obs_slice = c_obs_flat[idx_t]
                pred_slice = c_pred_flat[idx_t]
                if obs_slice.numel() < 2:
                    continue

                obs_max = torch.max(obs_slice).detach()
                obs_min = torch.min(obs_slice).detach()
                relief = (obs_max - obs_min) / torch.clamp(torch.abs(obs_max), min=1e-6)
                if float(relief.item()) < MULTI_HIGH_MIN_RELIEF:
                    continue

                high_cut = MULTI_HIGH_RATIO * obs_max
                high_mask = obs_slice >= high_cut
                if int(high_mask.sum().item()) < 2:
                    continue

                pred_max = torch.max(pred_slice).clamp_min(1e-6)
                obs_high_norm = obs_slice[high_mask] / torch.clamp(obs_max, min=1e-6)
                pred_high_norm = pred_slice[high_mask] / pred_max
                shape_loss = torch.mean((pred_high_norm - obs_high_norm) ** 2)

                low_mask = ~high_mask
                if torch.any(low_mask):
                    low_max = torch.max(pred_slice[low_mask])
                    high_min = torch.min(pred_slice[high_mask])
                    sep_loss = torch.relu(low_max + MULTI_HIGH_MARGIN - high_min)
                else:
                    sep_loss = torch.tensor(0.0, device=device)

                multi_high_shape_terms.append(shape_loss.detach())
                multi_high_sep_terms.append(sep_loss.detach())
                multi_high_losses.append(shape_loss + sep_loss)

            time_multi_losses = []
            for idx_s in station_time_indices:
                if idx_s.numel() < 2:
                    continue

                obs_seq = c_obs_flat[idx_s]
                pred_seq = c_pred_flat[idx_s]
                obs_station_max = torch.max(obs_seq).detach()
                obs_station_min = torch.min(obs_seq).detach()
                station_relief = (obs_station_max - obs_station_min) / torch.clamp(
                    torch.abs(obs_station_max), min=1e-6
                )
                if float(station_relief.item()) < MULTI_HIGH_MIN_RELIEF:
                    continue

                active_cut = MULTI_HIGH_RATIO * obs_station_max
                pair_active = torch.maximum(obs_seq[1:], obs_seq[:-1]) >= active_cut
                if not torch.any(pair_active):
                    continue

                obs_scale = torch.clamp(obs_station_max, min=1e-6)
                delta_obs = (obs_seq[1:] - obs_seq[:-1]) / obs_scale
                delta_pred = (pred_seq[1:] - pred_seq[:-1]) / obs_scale
                time_multi_losses.append(
                    torch.mean((delta_pred[pair_active] - delta_obs[pair_active]) ** 2)
                )

            if multi_high_losses or time_multi_losses:
                loss_terms = []
                if multi_high_losses:
                    loss_terms.append(torch.mean(torch.stack(multi_high_losses)))
                if time_multi_losses:
                    loss_terms.append(torch.mean(torch.stack(time_multi_losses)))
                loss_multi_high = torch.mean(torch.stack(loss_terms))
            else:
                loss_multi_high = torch.tensor(0.0, device=device)
            if multi_high_shape_terms:
                multi_high_shape_loss_dbg = torch.mean(torch.stack(multi_high_shape_terms))
            if multi_high_sep_terms:
                multi_high_sep_loss_dbg = torch.mean(torch.stack(multi_high_sep_terms))
            if time_multi_losses:
                multi_high_time_loss_dbg = torch.mean(torch.stack(time_multi_losses))
            multi_high_loss_cache = loss_multi_high.detach()
        elif LOSS_W_MULTI_HIGH > 0:
            loss_multi_high = multi_high_loss_cache
        else:
            loss_multi_high = torch.tensor(0.0, device=device)

        if LOSS_W_HIGH_DOWNWIND > 0 and aux_should_update:
            high_downwind_losses = []
            for i, idx_t in enumerate(obs_time_indices):
                if idx_t.numel() == 0:
                    continue
                obs_slice = c_obs_flat[idx_t]
                x_slice = x_obs_flat[idx_t]
                y_slice = y_obs_flat[idx_t]

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
            high_downwind_loss_cache = loss_high_downwind.detach()
        elif LOSS_W_HIGH_DOWNWIND > 0:
                loss_high_downwind = high_downwind_loss_cache
        else:
            loss_high_downwind = torch.tensor(0.0, device=device)
        sync_device()
        timing_obs_losses += time.perf_counter() - t_section

        # PDE residual
        sync_device()
        t_section = time.perf_counter()
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
        sync_device()
        timing_pde += time.perf_counter() - t_section

        # Extra physically constrained source-identification losses
        # 1) Source-local dominance: concentration at source center should exceed a nearby annulus.
        sync_device()
        t_section = time.perf_counter()
        if ENABLE_LOSS_SOURCE_LOCAL and LOSS_W_SOURCE_LOCAL > 0 and aux_should_update:
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
            n_time = t_w_t.numel()
            n_theta = theta.numel()
            ring_pts = torch.stack(
                [
                    ring_x.unsqueeze(0).repeat(n_time, 1).reshape(-1),
                    ring_y.unsqueeze(0).repeat(n_time, 1).reshape(-1),
                    t_w_t.view(-1, 1).repeat(1, n_theta).reshape(-1),
                ],
                dim=1,
            )
            ring_u = center_u.repeat_interleave(theta.numel(), dim=0)
            ring_v = center_v.repeat_interleave(theta.numel(), dim=0)
            c_ring = predict_concentration(model, ring_pts, ring_u, ring_v, SIGMA_SRC)
            loss_source_local = torch.relu(
                torch.mean(c_ring) + SOURCE_LOCAL_MARGIN - torch.mean(c_center)
            )
            source_local_loss_cache = loss_source_local.detach()
        elif ENABLE_LOSS_SOURCE_LOCAL and LOSS_W_SOURCE_LOCAL > 0:
            loss_source_local = source_local_loss_cache
        else:
            loss_source_local = torch.tensor(0.0, device=device)
        sync_device()
        timing_source_local += time.perf_counter() - t_section

        # 2) Wind-propagation corridor constraint (time-sliced, observation peaks):
        # observed high-value stations should lie in a reasonable downwind corridor
        # from the inferred source instead of only matching a predicted centroid.
        sync_device()
        t_section = time.perf_counter()
        if ENABLE_LOSS_AXIS and LOSS_W_AXIS > 0 and (epoch - 1) % axis_update_every == 0:
            axis_losses = []

            for i, idx_t in enumerate(obs_time_indices):
                if idx_t.numel() == 0:
                    continue

                obs_slice = c_obs_flat[idx_t]
                x_slice = x_obs_flat[idx_t]
                y_slice = y_obs_flat[idx_t]

                obs_max = torch.max(obs_slice).detach()
                obs_min = torch.min(obs_slice).detach()
                rel_contrast = (obs_max - obs_min) / torch.clamp(
                    torch.abs(obs_max), min=1e-6
                )
                if float(rel_contrast.item()) < AXIS_MIN_RELIEF:
                    continue

                high_cut = AXIS_HIGH_RATIO * obs_max
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
                along_high = dx_high * w_x + dy_high * w_y
                cross_high = torch.abs(-dx_high * w_y + dy_high * w_x)

                high_weight = torch.relu(obs_slice[high_mask]) / torch.clamp(
                    obs_max, min=1e-6
                )
                corridor_half_width = AXIS_CROSS_BASE + AXIS_CROSS_SLOPE * torch.relu(
                    along_high
                )
                forward_penalty = torch.relu(AXIS_ALONG_MARGIN - along_high)
                cross_penalty = torch.relu(cross_high - corridor_half_width)
                axis_losses.append(
                    torch.mean(high_weight * (forward_penalty + cross_penalty))
                )

            if len(axis_losses) > 0:
                loss_axis = torch.mean(torch.stack(axis_losses))
            else:
                loss_axis = torch.tensor(0.0, device=device)
            axis_loss_cache = loss_axis.detach()
        elif ENABLE_LOSS_AXIS and LOSS_W_AXIS > 0:
            loss_axis = axis_loss_cache
        else:
            loss_axis = torch.tensor(0.0, device=device)
        sync_device()
        timing_axis += time.perf_counter() - t_section

        # Two-stage schedule:
        # stage 1: prioritize fitting anomaly amplitudes / high-value stations
        # stage 2: smoothly restore full physics-consistent weighting
        if epoch <= STAGE1_EPOCHS:
            stage_blend = 0.0
        else:
            stage_blend = min(1.0, (epoch - STAGE1_EPOCHS) / max(1, PDE_RAMP_EPOCHS))

        curr_data_mult = STAGE1_DATA_MULT + (1.0 - STAGE1_DATA_MULT) * stage_blend
        curr_top_mult = STAGE1_TOP_STATION_MULT + (1.0 - STAGE1_TOP_STATION_MULT) * stage_blend
        curr_multi_mult = STAGE1_MULTI_HIGH_MULT + (1.0 - STAGE1_MULTI_HIGH_MULT) * stage_blend
        curr_high_downwind_mult = (
            STAGE1_HIGH_DOWNWIND_MULT
            + (1.0 - STAGE1_HIGH_DOWNWIND_MULT) * stage_blend
        )
        curr_source_local_mult = (
            STAGE1_SOURCE_LOCAL_MULT
            + (1.0 - STAGE1_SOURCE_LOCAL_MULT) * stage_blend
        )

        data_term = LOSS_W_DATA * curr_data_mult * loss_data
        pde_term = LOSS_W_PDE * loss_pde
        source_local_term = (
            LOSS_W_SOURCE_LOCAL * curr_source_local_mult * loss_source_local
        )
        axis_term = LOSS_W_AXIS * loss_axis
        top_station_term = LOSS_W_TOP_STATION * curr_top_mult * loss_top_station
        multi_high_term = LOSS_W_MULTI_HIGH * curr_multi_mult * loss_multi_high
        high_downwind_term = (
            LOSS_W_HIGH_DOWNWIND * curr_high_downwind_mult * loss_high_downwind
        )

        # Stage-aware PDE schedule:
        # early stage keeps PDE weak so the model first learns observation peaks,
        # then ramps to full physics after STAGE1_EPOCHS.
        if epoch <= STAGE1_EPOCHS:
            pde_factor = STAGE1_PDE_FACTOR
        else:
            pde_factor = STAGE1_PDE_FACTOR + (1.0 - STAGE1_PDE_FACTOR) * stage_blend
        pde_term_eff = pde_factor * pde_term

        raw_loss = (
            data_term
            + pde_term_eff
            + source_local_term
            + axis_term
            + top_station_term
            + multi_high_term
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
                + axis_term
                + top_station_term
                + multi_high_term
                + high_downwind_term
            )
        else:
            train_loss = raw_loss
            adaptive_weights = None

        sync_device()
        t_section = time.perf_counter()
        train_loss.backward()
        sync_device()
        timing_backward += time.perf_counter() - t_section

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

        sync_device()
        t_section = time.perf_counter()
        opt.step()
        if adaptive_opt is not None and epoch > adaptive_start_epoch:
            adaptive_opt.step()
        sync_device()
        timing_optimizer += time.perf_counter() - t_section

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
            if ENABLE_LOSS_AXIS and LOSS_W_AXIS > 0:
                loss_parts.append(f"axis={loss_axis.item():.4f}")
            if LOSS_W_TOP_STATION > 0:
                loss_parts.append(f"top_station={loss_top_station.item():.4f}")
            if LOSS_W_MULTI_HIGH > 0:
                loss_parts.append(f"multi_high={loss_multi_high.item():.4f}")
            if LOSS_W_HIGH_DOWNWIND > 0:
                loss_parts.append(f"high_downwind={loss_high_downwind.item():.4f}")
            print(
                f"Epoch {epoch}: raw_loss={raw_loss.item():4f}, "
                f"{', '.join(loss_parts)}, "
                f"D={D.item():.3e}, Q_mean={Q_mean.item():.4f}, "
                f"xs={xs.item():.3f}, ys={ys.item():.3f}, pde_factor={pde_factor:.3f}, "
                f"data_mult={curr_data_mult:.2f}, multi_mult={curr_multi_mult:.2f}"
                f"{extra}"
            )
            sync_device()
            epoch_total_time = time.perf_counter() - epoch_start_time
            print(
                "Timing: "
                f"data_forward={timing_data_forward:.3f}s, "
                f"obs_losses={timing_obs_losses:.3f}s, "
                f"pde={timing_pde:.3f}s, "
                f"source_local={timing_source_local:.3f}s, "
                f"axis={timing_axis:.3f}s, "
                f"backward={timing_backward:.3f}s, "
                f"optimizer={timing_optimizer:.3f}s, "
                f"epoch_total={epoch_total_time:.3f}s"
            )
        if DEBUG_EVERY > 0 and epoch % DEBUG_EVERY == 0:
            if TRAIN_ON_RESIDUAL:
                pred_raw = c_pred * c_scale + c_obs_baseline_t
            else:
                pred_raw = c_pred * c_scale
            pred_raw_flat = pred_raw.detach().view(-1)
            obs_raw_flat = c_obs_raw_t.detach().view(-1)
            baseline_flat = c_obs_baseline_t.detach().view(-1)
            obs_fit_flat = c_obs_t.detach().view(-1)
            gate_flat = gate_obs.detach().view(-1)

            peak_hit_values = []
            peak_obs_values = []
            peak_pred_values = []
            peak_gate_values = []
            peak_q_values = []
            peak_plume_values = []
            multi_station_counts = []
            for idx_t in obs_time_indices:
                if idx_t.numel() == 0:
                    continue
                obs_slice_fit = obs_fit_flat[idx_t]
                pred_slice_raw = pred_raw_flat[idx_t]
                obs_slice_raw = obs_raw_flat[idx_t]
                gate_slice = gate_flat[idx_t]
                q_slice = q_obs.detach().view(-1)[idx_t]
                plume_slice = plume_obs.detach().view(-1)[idx_t]

                top_idx = int(torch.argmax(obs_slice_fit).item())
                pred_top = pred_slice_raw[top_idx]
                obs_top = obs_slice_raw[top_idx]
                peak_hit_values.append(
                    (pred_top / torch.clamp(obs_top, min=1e-6)).detach()
                )
                peak_obs_values.append(obs_top.detach())
                peak_pred_values.append(pred_top.detach())
                peak_gate_values.append(gate_slice[top_idx].detach())
                peak_q_values.append(q_slice[top_idx].detach())
                peak_plume_values.append(plume_slice[top_idx].detach())

                obs_max_fit = torch.max(obs_slice_fit)
                relief = (obs_max_fit - torch.min(obs_slice_fit)) / torch.clamp(
                    torch.abs(obs_max_fit), min=1e-6
                )
                if float(relief.item()) >= MULTI_HIGH_MIN_RELIEF:
                    high_mask = obs_slice_fit >= (MULTI_HIGH_RATIO * obs_max_fit)
                    multi_station_counts.append(float(high_mask.sum().item()))

            if peak_hit_values:
                peak_hit_mean = torch.mean(torch.stack(peak_hit_values)).item()
                peak_obs_mean = torch.mean(torch.stack(peak_obs_values)).item()
                peak_pred_mean = torch.mean(torch.stack(peak_pred_values)).item()
                peak_gate_mean = torch.mean(torch.stack(peak_gate_values)).item()
                peak_q_mean_dbg = torch.mean(torch.stack(peak_q_values)).item()
                peak_plume_mean = torch.mean(torch.stack(peak_plume_values)).item()
            else:
                peak_hit_mean = 0.0
                peak_obs_mean = 0.0
                peak_pred_mean = 0.0
                peak_gate_mean = 0.0
                peak_q_mean_dbg = 0.0
                peak_plume_mean = 0.0

            if multi_station_counts:
                multi_station_count_mean = float(np.mean(multi_station_counts))
                multi_station_count_max = float(np.max(multi_station_counts))
            else:
                multi_station_count_mean = 0.0
                multi_station_count_max = 0.0
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
                        f"baseline_mean={baseline_flat.mean().item():.4f}",
                        f"obs_fit_mean={obs_fit_flat.mean().item():.4f}",
                        f"pred_raw_mean={pred_raw_flat.mean().item():.4f}",
                        f"pred_raw_max={pred_raw_flat.max().item():.4f}",
                        f"peak_hit_mean={peak_hit_mean:.4f}",
                        f"peak_obs_mean={peak_obs_mean:.4f}",
                        f"peak_pred_mean={peak_pred_mean:.4f}",
                        f"peak_gate_mean={peak_gate_mean:.4f}",
                        f"peak_q_obs_mean={peak_q_mean_dbg:.4f}",
                        f"peak_plume_mean={peak_plume_mean:.4f}",
                        f"multi_high_shape_dbg={safe_scalar(multi_high_shape_loss_dbg):.4f}",
                        f"multi_high_sep_dbg={safe_scalar(multi_high_sep_loss_dbg):.4f}",
                        f"multi_high_time_dbg={safe_scalar(multi_high_time_loss_dbg):.4f}",
                        f"multi_station_count_mean={multi_station_count_mean:.2f}",
                        f"multi_station_count_max={multi_station_count_max:.2f}",
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

    xyt_diag = torch.cat([x_obs_t, y_obs_t, t_obs_t], dim=1)
    with torch.no_grad():
        pred_diag = predict_concentration(model, xyt_diag, u_obs_t, v_obs_t, SIGMA_SRC)
        if TRAIN_ON_RESIDUAL:
            pred_diag_raw = pred_diag * c_scale + c_obs_baseline_t
        else:
            pred_diag_raw = pred_diag * c_scale

    plot_station_timeseries(
        times=np.array(obs_time_labels, dtype="datetime64[ns]"),
        station_names=np.array(obs_station_labels, dtype=object),
        obs_values=c_obs_raw,
        pred_values=pred_diag_raw.detach().cpu().numpy().reshape(-1),
        title="Observed vs Predicted Concentration by Station",
    )

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
