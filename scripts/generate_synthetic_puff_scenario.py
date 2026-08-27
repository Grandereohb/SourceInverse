"""Generate one truth-known source-inversion scenario using Gaussian puffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from data_io import load_sites, load_wind, wind_dir_to_uv  # noqa: E402
from geo_utils import xy_to_latlon  # noqa: E402
from synthetic_puff import (  # noqa: E402
    gaussian_puff_station_concentrations,
    source_strength_shape,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--wind", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario-id", default="dev_puff_0001")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--source-fraction-x", type=float, default=0.68)
    parser.add_argument("--source-fraction-y", type=float, default=0.32)
    parser.add_argument("--source-pad-m", type=float, default=500.0)
    parser.add_argument("--q-shape", default="single_pulse")
    parser.add_argument("--target-signal-peak", type=float, default=120.0)
    parser.add_argument("--diffusivity-m2s", type=float, default=8.0)
    parser.add_argument("--initial-sigma-m", type=float, default=120.0)
    parser.add_argument("--decay-per-hour", type=float, default=0.08)
    parser.add_argument("--truth-wind-factor", type=float, default=0.35)
    parser.add_argument("--pre-event-hours", type=float, default=1.0)
    parser.add_argument("--integration-dt-h", type=float, default=0.02)
    parser.add_argument("--background-level", type=float, default=3.0)
    parser.add_argument("--station-bias-fraction", type=float, default=0.10)
    parser.add_argument("--noise-absolute", type=float, default=0.35)
    parser.add_argument("--noise-relative", type=float, default=0.04)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sites, lon0, lat0 = load_sites(args.sites)
    wind = load_wind(args.wind).sort_values("time").reset_index(drop=True)
    observation_times_h = (
        (wind["time"] - wind["time"].iloc[0]).dt.total_seconds().to_numpy()
        / 3600.0
    )
    wind_u, wind_v = wind_dir_to_uv(
        wind["dir"].to_numpy(dtype=np.float64),
        wind["sp"].to_numpy(dtype=np.float64),
        is_from=True,
    )
    x_min = float(sites["x"].min() - args.source_pad_m)
    x_max = float(sites["x"].max() + args.source_pad_m)
    y_min = float(sites["y"].min() - args.source_pad_m)
    y_max = float(sites["y"].max() + args.source_pad_m)
    source_x = x_min + float(args.source_fraction_x) * (x_max - x_min)
    source_y = y_min + float(args.source_fraction_y) * (y_max - y_min)
    source_lon, source_lat = xy_to_latlon(source_x, source_y, lon0, lat0)

    q_reference_start = float(observation_times_h[0])
    q_reference_duration = max(
        float(observation_times_h[-1] - observation_times_h[0]), 1.0
    )

    def q_unit(times):
        return source_strength_shape(
            times,
            args.q_shape,
            reference_start_h=q_reference_start,
            reference_duration_h=q_reference_duration,
        )

    unit_signal = gaussian_puff_station_concentrations(
        station_x_m=sites["x"],
        station_y_m=sites["y"],
        observation_times_h=observation_times_h,
        wind_times_h=observation_times_h,
        wind_u_mps=wind_u,
        wind_v_mps=wind_v,
        source_x_m=source_x,
        source_y_m=source_y,
        q_at_time=q_unit,
        diffusivity_m2s=args.diffusivity_m2s,
        initial_sigma_m=args.initial_sigma_m,
        decay_per_hour=args.decay_per_hour,
        wind_factor=args.truth_wind_factor,
        pre_event_hours=args.pre_event_hours,
        integration_dt_h=args.integration_dt_h,
    )
    unit_peak = max(float(np.max(unit_signal)), 1e-12)
    q_amplitude = float(args.target_signal_peak) / unit_peak

    def q_truth(times):
        return q_amplitude * q_unit(times)

    signal = unit_signal * q_amplitude
    phase = observation_times_h / max(float(observation_times_h[-1]), 1.0)
    common_background = float(args.background_level) * (
        1.0 + 0.15 * np.sin(2.0 * np.pi * phase)
    )
    station_bias = rng.normal(
        0.0,
        float(args.station_bias_fraction) * float(args.background_level),
        size=len(sites),
    )
    background = common_background[:, None] + station_bias[None, :]
    noiseless = np.maximum(signal + background, 0.0)
    noise_scale = float(args.noise_absolute) + float(args.noise_relative) * np.sqrt(
        np.maximum(noiseless, 0.0)
    )
    observed = np.maximum(noiseless + rng.normal(0.0, noise_scale), 0.0)

    concentration = pd.DataFrame({"time": wind["time"]})
    for index, station in enumerate(sites["station"]):
        concentration[str(station)] = observed[:, index]
    concentration["TARGET_POLLUTANT"] = f"synthetic_{args.scenario_id}"
    concentration_path = output_dir / "concentration.csv"
    concentration.to_csv(concentration_path, index=False, encoding="utf-8-sig")

    long_rows = []
    for time_index, timestamp in enumerate(wind["time"]):
        for station_index, station in enumerate(sites["station"]):
            long_rows.append(
                {
                    "scenario_id": args.scenario_id,
                    "time": str(timestamp),
                    "time_h": float(observation_times_h[time_index]),
                    "station": str(station),
                    "station_x_m": float(sites["x"].iloc[station_index]),
                    "station_y_m": float(sites["y"].iloc[station_index]),
                    "signal": float(signal[time_index, station_index]),
                    "background": float(background[time_index, station_index]),
                    "noiseless": float(noiseless[time_index, station_index]),
                    "observed": float(observed[time_index, station_index]),
                }
            )
    truth_station_path = output_dir / "truth_station_concentrations.csv"
    pd.DataFrame(long_rows).to_csv(
        truth_station_path, index=False, encoding="utf-8-sig"
    )

    sites_copy = output_dir / f"sites{args.sites.suffix.lower()}"
    wind_copy = output_dir / f"wind{args.wind.suffix.lower()}"
    shutil.copy2(args.sites, sites_copy)
    shutil.copy2(args.wind, wind_copy)
    manifest = {
        "schema_version": 1,
        "scenario_id": args.scenario_id,
        "split": "development",
        "truth_model": "independent_gaussian_puff_fine_time_integration",
        "random_seed": args.seed,
        "source": {
            "x_m": source_x,
            "y_m": source_y,
            "lon": float(source_lon),
            "lat": float(source_lat),
            "fraction_x": float(args.source_fraction_x),
            "fraction_y": float(args.source_fraction_y),
        },
        "source_strength": {
            "shape": args.q_shape,
            "amplitude": q_amplitude,
            "values_at_observation_times": [
                float(value) for value in q_truth(observation_times_h)
            ],
            "units": "arbitrary concentration-area per hour",
        },
        "physics": {
            "diffusivity_m2s": float(args.diffusivity_m2s),
            "initial_sigma_m": float(args.initial_sigma_m),
            "decay_per_hour": float(args.decay_per_hour),
            "truth_wind_factor": float(args.truth_wind_factor),
            "pre_event_hours": float(args.pre_event_hours),
            "integration_dt_h": float(args.integration_dt_h),
        },
        "observation_model": {
            "background_level": float(args.background_level),
            "station_bias_fraction": float(args.station_bias_fraction),
            "noise_absolute": float(args.noise_absolute),
            "noise_relative": float(args.noise_relative),
            "realized_signal_peak": float(np.max(signal)),
            "realized_observed_peak": float(np.max(observed)),
        },
        "files": {},
    }
    for label, path in {
        "sites": sites_copy,
        "wind": wind_copy,
        "concentration": concentration_path,
        "truth_station_concentrations": truth_station_path,
    }.items():
        manifest["files"][label] = {"path": str(path), "sha256": _sha256(path)}
    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"scenario={args.scenario_id}")
    print(f"output_dir={output_dir}")
    print(f"source_xy_m=({source_x:.3f}, {source_y:.3f})")
    print(f"manifest_sha256={_sha256(manifest_path)}")


if __name__ == "__main__":
    main()
