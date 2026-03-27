import os
import math
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ---------- Config ----------
SITE_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\zhenhua\sites.xlsx"
CONC_PATH = (
    r"C:\Document\phd\SourceInverse\SourceInverse\data\zhenhua\concentration.xlsx"
)
WIND_PATH = r"C:\Document\phd\SourceInverse\SourceInverse\data\zhenhua\wind.xlsx"

# If your wind direction is "from" (meteorological), keep True.
# If your dir is "to" (where wind is blowing toward), set False.
WIND_DIR_IS_FROM = True

# Training hyperparams
EPOCHS = 5000
LR = 1e-3
N_COLLOCATION = 5000

# ---------- Utilities ----------


def dms_to_decimal(dms_str: str) -> float:
    """Convert DMS to decimal degrees."""
    if dms_str is None:
        raise ValueError("Invalid DMS format: None")

    # If already numeric, return directly
    if isinstance(dms_str, (int, float, np.floating)):
        if np.isnan(dms_str):
            raise ValueError("Invalid DMS format: NaN")
        return float(dms_str)

    s = str(dms_str).strip()

    # If it's already a plain decimal string, return directly
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        return float(s)

    # Normalize degree symbol variants / mojibake
    s = s.replace("\\u00c2\\u00b0", "\\u00b0")  # mojibake for degree
    s = s.replace("\\u040e\\u0433", "\\u00b0")  # mojibake for degree
    for deg_sym in ["\u63b3", "\u00ba", "\u02da", "\u00b0"]:
        s = s.replace(deg_sym, "\u00b0")
    s = s.replace("\u2032", "'").replace("\u2033", '"')

    # Normalize full-width digits and symbols
    s = s.translate(
        str.maketrans(
            "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19\uff0e\uff0d\uff0b",
            "0123456789.-+",
        )
    )

    # Extract numbers robustly
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", s)]
    if len(nums) >= 3:
        deg, minute, sec = nums[0], nums[1], nums[2]
        return deg + minute / 60.0 + sec / 3600.0
    if len(nums) == 2:
        deg, minute = nums[0], nums[1]
        return deg + minute / 60.0
    if len(nums) == 1:
        return nums[0]
    raise ValueError(f"Invalid DMS format: {dms_str}")


def latlon_to_xy(lon, lat, lon0, lat0):
    """Local tangent plane in meters."""
    # meters per degree
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


def xy_to_latlon(x, y, lon0, lat0):
    """Inverse of local tangent plane projection."""
    lon = lon0 + x / (math.cos(math.radians(lat0)) * 111320.0)
    lat = lat0 + y / 110540.0
    return lon, lat


def load_sites(path):
    df = pd.read_excel(path)
    # Two supported formats:
    # A) columns: station, lon, lat
    # B) columns: station, N, S, E ... with rows [lon, lat]
    cols = {str(c).lower(): c for c in df.columns}

    if "station" in cols and any(
        str(v).lower() in ["lon", "lat"] for v in df[cols["station"]].tolist()
    ):
        # Format B
        st_col = cols["station"]
        station_cols = [c for c in df.columns if c != st_col]
        df2 = df.set_index(st_col)
        lons = [dms_to_decimal(df2.loc["lon", c]) for c in station_cols]
        lats = [dms_to_decimal(df2.loc["lat", c]) for c in station_cols]
        stations = [str(c) for c in station_cols]
    else:
        # Format A
        st_col = cols.get("station", df.columns[0])
        lon_col = cols.get("lon", df.columns[1])
        lat_col = cols.get("lat", df.columns[2])
        stations = df[st_col].astype(str).tolist()
        lons = [dms_to_decimal(v) for v in df[lon_col].tolist()]
        lats = [dms_to_decimal(v) for v in df[lat_col].tolist()]

    lon0 = float(np.mean(lons))
    lat0 = float(np.mean(lats))
    xy = [latlon_to_xy(lon, lat, lon0, lat0) for lon, lat in zip(lons, lats)]
    site_df = pd.DataFrame(
        {
            "station": stations,
            "lon": lons,
            "lat": lats,
            "x": [p[0] for p in xy],
            "y": [p[1] for p in xy],
        }
    )
    return site_df, lon0, lat0


def load_wind(path):
    df = pd.read_excel(path)
    # Expect columns: time, dir, sp (first column is time)
    cols = {str(c).lower(): c for c in df.columns}
    t_col = df.columns[0]
    dir_col = cols.get("dir", df.columns[1])
    sp_col = cols.get("sp", df.columns[2])
    out = df[[t_col, dir_col, sp_col]].copy()
    out.columns = ["time", "dir", "sp"]
    out["time"] = pd.to_datetime(out["time"])
    return out


def load_conc(path):
    df = pd.read_excel(path)
    # Expect columns: time, N, E, S (or other station labels); first column is time
    t_col = df.columns[0]
    out = df.copy()
    out = out.rename(columns={t_col: "time"})
    out["time"] = pd.to_datetime(out["time"])
    return out


def wind_dir_to_uv(dir_deg, sp, is_from=True):
    # dir_deg: 0 = North, 90 = East, clockwise
    rad = np.deg2rad(dir_deg)
    if is_from:
        u = -sp * np.sin(rad)
        v = -sp * np.cos(rad)
    else:
        u = sp * np.sin(rad)
        v = sp * np.cos(rad)
    return u, v


# ---------- PINN Model ----------
class PINN(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Learnable physical parameters
        self.logD = nn.Parameter(torch.tensor(0.0))  # D = exp(logD)
        self.logQ = nn.Parameter(torch.tensor(0.0))  # Q = exp(logQ)
        self.xs = nn.Parameter(torch.tensor(0.0))
        self.ys = nn.Parameter(torch.tensor(0.0))

    def forward(self, xyt):
        return self.net(xyt)

    def D(self):
        return torch.exp(self.logD)

    def Q(self):
        return torch.exp(self.logQ)


# ---------- Training ----------


def main():
    sites, lon0, lat0 = load_sites(SITE_PATH)
    wind = load_wind(WIND_PATH)
    conc = load_conc(CONC_PATH)

    # Merge on time
    data = pd.merge(conc, wind, on="time", how="inner")

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
    # For each time, we create a sample for each station
    obs = []
    for _, row in data.iterrows():
        u, v = wind_dir_to_uv(row["dir"], row["sp"], is_from=WIND_DIR_IS_FROM)
        for _, srow in sites.iterrows():
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
    model = PINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_obs_t = torch.tensor(x_obs, dtype=torch.float32, device=device).view(-1, 1)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=device).view(-1, 1)
    t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)
    c_obs_t = torch.tensor(c_obs, dtype=torch.float32, device=device).view(-1, 1)
    # u_obs_t = torch.tensor(u_obs, dtype=torch.float32, device=device).view(-1, 1)
    # v_obs_t = torch.tensor(v_obs, dtype=torch.float32, device=device).view(-1, 1)

    # Pre-generate collocation points
    rng = np.random.default_rng(0)
    x_col = rng.uniform(x_min, x_max, N_COLLOCATION)
    y_col = rng.uniform(y_min, y_max, N_COLLOCATION)
    t_col = rng.uniform(t_min, t_max, N_COLLOCATION)

    x_col_t = torch.tensor(x_col, dtype=torch.float32, device=device).view(-1, 1)
    y_col_t = torch.tensor(y_col, dtype=torch.float32, device=device).view(-1, 1)
    t_col_t = torch.tensor(t_col, dtype=torch.float32, device=device).view(-1, 1)

    # For collocation points, use nearest wind values by time
    # Simpler: interpolate by time from observation wind
    # Build a simple linear interpolation in numpy
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
        # This avoids a delta function while still allowing source localization.
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

    # --------- Visualization ---------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(sites_plot["lon"], sites_plot["lat"], c="blue", s=80, label="Stations")
    for _, r in sites_plot.iterrows():
        plt.text(
            r["lon"], r["lat"], str(r["station"]), fontsize=10, ha="left", va="bottom"
        )
    plt.scatter(
        pred_lon, pred_lat, c="red", s=150, marker="*", label="Estimated Source"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Stations and Estimated Source")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------- Diffusion Animation (better scaling) ---------
    from matplotlib import animation

    model.eval()
    nx, ny = 120, 120
    n_frames = 40
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
        xyt_t = torch.tensor(xyt, dtype=torch.float32, device=device)
        with torch.no_grad():
            cc = model(xyt_t).cpu().numpy().reshape(ny, nx)
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
    ax.scatter(pred_lon, pred_lat, c="red", s=80, marker="*", label="Estimated Source")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Concentration")

    def frame_fn(i):
        im.set_data(frames[i])
        ax.set_title(f"Diffusion Over Time (t={t_frames[i]:.2f} h)")
        return [im]

    ani = animation.FuncAnimation(
        fig, frame_fn, frames=n_frames, interval=200, blit=False
    )
    ani.save("diffusion.gif", writer="pillow", fps=5)
    plt.show()


if __name__ == "__main__":
    main()
