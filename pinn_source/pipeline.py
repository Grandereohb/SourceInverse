import math
import numpy as np
import torch

from config import EPOCHS, LR, N_COLLOCATION, WIND_DIR_IS_FROM, MODEL_NAME
from data_io import load_sites, load_wind, load_conc, wind_dir_to_uv
from model_registry import get_model
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

    # Collocation points for PDE residual
    x_min, x_max = sites["x"].min(), sites["x"].max()
    y_min, y_max = sites["y"].min(), sites["y"].max()
    t_min, t_max = np.min(t_obs), np.max(t_obs)

    # Expand domain a bit
    pad = 50.0
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    # Build tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelCls = get_model(MODEL_NAME)
    model = ModelCls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_obs_t = torch.tensor(x_obs, dtype=torch.float32, device=device).view(-1, 1)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=device).view(-1, 1)
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)
    c_obs_t = torch.tensor(c_obs, dtype=torch.float32, device=device).view(-1, 1)

    # Pre-generate collocation points
    rng = np.random.default_rng(0)
    x_col = rng.uniform(x_min, x_max, N_COLLOCATION)
    y_col = rng.uniform(y_min, y_max, N_COLLOCATION)
    t_col = rng.uniform(t_min, t_max, N_COLLOCATION)

    x_col_t = torch.tensor(x_col, dtype=torch.float32, device=device).view(-1, 1)
    y_col_t = torch.tensor(y_col, dtype=torch.float32, device=device).view(-1, 1)
    t_col_t = torch.tensor(t_col, dtype=torch.float32, device=device).view(-1, 1)

    # For collocation points, use nearest wind values by time
    t_w = np.unique(t_obs)
    u_w = []
    v_w = []
    for tw in t_w:
        idx = np.where(t_obs == tw)[0][0]
        u_w.append(u_obs[idx])
        v_w.append(v_obs[idx])
    u_w = np.array(u_w)
    v_w = np.array(v_w)

    u_col = np.interp(t_col, t_w, u_w)
    v_col = np.interp(t_col, t_w, v_w)
    u_col_t = torch.tensor(u_col, dtype=torch.float32, device=device).view(-1, 1)
    v_col_t = torch.tensor(v_col, dtype=torch.float32, device=device).view(-1, 1)

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()

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
        c_yy = torch.autograd.grad(
            c_y, xyt_col, torch.ones_like(c_y), create_graph=True
        )[0][:, 1:2]

        D = model.D()
        # Advection term with time-varying wind (u,v)
        u_c = u_col_t
        v_c = v_col_t

        # Source term: approximate as a narrow Gaussian centered at (xs, ys)
        sigma_src = 20.0
        xs = model.xs
        ys = model.ys
        src = model.Q() * torch.exp(
            -((xyt_col[:, 0:1] - xs) ** 2 + (xyt_col[:, 1:2] - ys) ** 2)
            / (2 * sigma_src**2)
        )

        residual = c_t + u_c * c_x + v_c * c_y - D * (c_xx + c_yy) - src
        loss_pde = torch.mean(residual**2)

        # Regularization to keep xs, ys inside domain
        penalty = 0.0
        penalty += torch.relu(x_min - xs) ** 2 + torch.relu(xs - x_max) ** 2
        penalty += torch.relu(y_min - ys) ** 2 + torch.relu(ys - y_max) ** 2

        loss = loss_data + 0.1 * loss_pde + 0.01 * penalty
        loss.backward()
        opt.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}: loss={loss.item():.6f}, D={D.item():.4f}, Q={model.Q().item():.4f}, xs={xs.item():.2f}, ys={ys.item():.2f}"
            )

    # Convert predicted source back to lat/lon
    xs = model.xs.item()
    ys = model.ys.item()
    pred_lon = lon0 + xs / (math.cos(math.radians(lat0)) * 111320.0)
    pred_lat = lat0 + ys / 110540.0

    print("Estimated source (x,y) meters:", xs, ys)
    print("Estimated source (lat,lon):", pred_lat, pred_lon)

    plot_sites_and_source(sites_plot, pred_lon, pred_lat)

    diffusion_animation(
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
    )
