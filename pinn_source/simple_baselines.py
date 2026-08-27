"""Transparent no-optimization baselines for source localization."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data_io import load_conc, load_sites, load_wind, wind_dir_to_uv


def prepare_station_anomalies(site_path, concentration_path, wind_path):
    sites, lon0, lat0 = load_sites(site_path)
    concentration = load_conc(concentration_path)
    wind = load_wind(wind_path)
    merged = concentration.merge(wind, on="time", how="inner")
    station_columns = [
        station
        for station in sites["station"].astype(str)
        if station in merged.columns and not merged[station].isna().all()
    ]
    if not station_columns:
        raise ValueError("No station columns found in concentration data.")
    merged = merged.dropna(subset=["dir", "sp"] + station_columns).copy()
    if merged.empty:
        raise ValueError("No complete timestamps remain for baseline estimation.")
    values = merged[station_columns].astype(float)
    background = values.median(axis=1)
    anomaly = values.subtract(background, axis=0).clip(lower=0.0)
    return sites, lon0, lat0, merged, anomaly


def maximum_anomaly_station(site_path, concentration_path, wind_path) -> dict:
    sites, _, _, merged, anomaly = prepare_station_anomalies(
        site_path, concentration_path, wind_path
    )
    flat_index = int(np.nanargmax(anomaly.to_numpy(dtype=np.float64)))
    time_index, station_index = np.unravel_index(flat_index, anomaly.shape)
    station_name = str(anomaly.columns[station_index])
    station_row = sites.loc[sites["station"].astype(str) == station_name].iloc[0]
    return {
        "method": "B0_maximum_anomaly_station",
        "source_x_m": float(station_row["x"]),
        "source_y_m": float(station_row["y"]),
        "peak_station": station_name,
        "peak_time": str(merged.iloc[time_index]["time"]),
        "peak_anomaly": float(anomaly.iloc[time_index, station_index]),
        "peak_wind_direction_deg": float(merged.iloc[time_index]["dir"]),
        "peak_wind_speed_mps": float(merged.iloc[time_index]["sp"]),
    }


def maximum_anomaly_upwind(
    site_path,
    concentration_path,
    wind_path,
    *,
    distance_m: float,
    source_position_pad_m: float,
) -> dict:
    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and non-negative.")
    if not math.isfinite(source_position_pad_m):
        raise ValueError("source_position_pad_m must be finite.")
    sites, _, _, merged, anomaly = prepare_station_anomalies(
        site_path, concentration_path, wind_path
    )
    b0 = maximum_anomaly_station(site_path, concentration_path, wind_path)
    peak_row = merged.loc[merged["time"].astype(str) == b0["peak_time"]]
    if peak_row.empty:
        raise RuntimeError("Peak timestamp disappeared during baseline calculation.")
    u, v = wind_dir_to_uv(
        np.asarray([float(peak_row.iloc[0]["dir"])]),
        np.asarray([float(peak_row.iloc[0]["sp"])]),
        is_from=True,
    )
    speed = float(math.hypot(float(u[0]), float(v[0])))
    if speed <= 1e-12:
        requested_x, requested_y = b0["source_x_m"], b0["source_y_m"]
    else:
        requested_x = b0["source_x_m"] - distance_m * float(u[0]) / speed
        requested_y = b0["source_y_m"] - distance_m * float(v[0]) / speed
    x_min = float(sites["x"].min() - source_position_pad_m)
    x_max = float(sites["x"].max() + source_position_pad_m)
    y_min = float(sites["y"].min() - source_position_pad_m)
    y_max = float(sites["y"].max() + source_position_pad_m)
    source_x = float(np.clip(requested_x, x_min, x_max))
    source_y = float(np.clip(requested_y, y_min, y_max))
    return {
        **b0,
        "method": "B1_maximum_anomaly_station_fixed_upwind_shift",
        "source_x_m": source_x,
        "source_y_m": source_y,
        "upwind_distance_m": float(distance_m),
        "unclipped_source_x_m": float(requested_x),
        "unclipped_source_y_m": float(requested_y),
        "was_clipped": bool(source_x != requested_x or source_y != requested_y),
        "source_domain_m": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "wind_usage": "raw wind vector at maximum-anomaly timestamp",
    }
