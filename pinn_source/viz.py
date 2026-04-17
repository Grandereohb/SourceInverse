import numpy as np
import torch
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt

from geo_utils import xy_to_latlon
from config import VISUALIZE_GATE_ONLY, ADD_BASELINE_TO_VIZ
from field import predict_concentration, field_components

# Font config for CJK labels
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False


def plot_sites_and_source(sites_plot, pred_lon, pred_lat):
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
    plt.show()
