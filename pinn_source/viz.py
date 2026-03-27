import numpy as np
import torch
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt

from geo_utils import xy_to_latlon

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
    ani.save(out_gif, writer="pillow", fps=5)
    plt.show()
