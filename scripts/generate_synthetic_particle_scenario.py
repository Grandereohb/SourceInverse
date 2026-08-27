"""Generate a truth-known scenario with an independent particle receptor model."""

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
from synthetic_particle import lagrangian_particle_station_concentrations  # noqa: E402
from synthetic_puff import source_strength_shape  # noqa: E402


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--wind", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-fraction-x", required=True, type=float)
    parser.add_argument("--source-fraction-y", required=True, type=float)
    parser.add_argument("--source-pad-m", type=float, default=500.0)
    parser.add_argument("--q-shape", default="double_peak")
    parser.add_argument("--target-signal-peak", type=float, default=100.0)
    parser.add_argument("--diffusivity-m2s", type=float, default=2.0)
    parser.add_argument("--initial-sigma-m", type=float, default=329.3)
    parser.add_argument("--decay-per-hour", type=float, default=0.5)
    parser.add_argument("--truth-wind-factor", type=float, default=0.25)
    parser.add_argument("--diffusivity-along-m2s", type=float)
    parser.add_argument("--diffusivity-cross-m2s", type=float)
    parser.add_argument("--initial-sigma-along-m", type=float)
    parser.add_argument("--initial-sigma-cross-m", type=float)
    parser.add_argument("--sensor-kernel-sigma-m", type=float, default=60.0)
    parser.add_argument("--meander-amplitude-m", type=float, default=0.0)
    parser.add_argument("--meander-period-h", type=float, default=6.0)
    parser.add_argument("--particles-per-release", type=int, default=64)
    parser.add_argument("--release-dt-h", type=float, default=0.1)
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
    times_h = (wind["time"] - wind["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    wind_u, wind_v = wind_dir_to_uv(wind["dir"].to_numpy(float), wind["sp"].to_numpy(float), is_from=True)
    x_min = float(sites["x"].min() - args.source_pad_m)
    x_max = float(sites["x"].max() + args.source_pad_m)
    y_min = float(sites["y"].min() - args.source_pad_m)
    y_max = float(sites["y"].max() + args.source_pad_m)
    source_x = x_min + args.source_fraction_x * (x_max - x_min)
    source_y = y_min + args.source_fraction_y * (y_max - y_min)
    source_lon, source_lat = xy_to_latlon(source_x, source_y, lon0, lat0)
    duration = max(float(times_h[-1] - times_h[0]), 1.0)

    def q_unit(times):
        return source_strength_shape(times, args.q_shape, float(times_h[0]), duration)

    along_d = args.diffusivity_m2s if args.diffusivity_along_m2s is None else args.diffusivity_along_m2s
    cross_d = args.diffusivity_m2s if args.diffusivity_cross_m2s is None else args.diffusivity_cross_m2s
    along_sigma = args.initial_sigma_m if args.initial_sigma_along_m is None else args.initial_sigma_along_m
    cross_sigma = args.initial_sigma_m if args.initial_sigma_cross_m is None else args.initial_sigma_cross_m
    particle_settings = dict(
        station_x_m=sites["x"], station_y_m=sites["y"], observation_times_h=times_h,
        wind_times_h=times_h, wind_u_mps=wind_u, wind_v_mps=wind_v,
        source_x_m=source_x, source_y_m=source_y, q_at_time=q_unit,
        diffusivity_along_m2s=along_d, diffusivity_cross_m2s=cross_d,
        initial_sigma_along_m=along_sigma, initial_sigma_cross_m=cross_sigma,
        sensor_kernel_sigma_m=args.sensor_kernel_sigma_m, wind_factor=args.truth_wind_factor,
        decay_per_hour=args.decay_per_hour, meander_amplitude_m=args.meander_amplitude_m,
        meander_period_h=args.meander_period_h, pre_event_hours=args.pre_event_hours,
        release_dt_h=args.release_dt_h, particles_per_release=args.particles_per_release,
        random_seed=args.seed,
    )
    unit_signal = lagrangian_particle_station_concentrations(**particle_settings)
    amplitude = float(args.target_signal_peak) / max(float(unit_signal.max()), 1e-12)
    signal = unit_signal * amplitude
    phase = times_h / max(float(times_h[-1]), 1.0)
    common_background = args.background_level * (1.0 + 0.15 * np.sin(2.0 * np.pi * phase))
    station_bias = rng.normal(0.0, args.station_bias_fraction * args.background_level, size=len(sites))
    background = common_background[:, None] + station_bias[None, :]
    noiseless = np.maximum(signal + background, 0.0)
    noise_scale = args.noise_absolute + args.noise_relative * np.sqrt(noiseless)
    observed = np.maximum(noiseless + rng.normal(0.0, noise_scale), 0.0)
    concentration = pd.DataFrame({"time": wind["time"]})
    for index, station in enumerate(sites["station"]):
        concentration[str(station)] = observed[:, index]
    concentration["TARGET_POLLUTANT"] = f"synthetic_{args.scenario_id}"
    concentration_path = output_dir / "concentration.csv"
    concentration.to_csv(concentration_path, index=False, encoding="utf-8-sig")
    long_rows = []
    for ti, timestamp in enumerate(wind["time"]):
        for si, station in enumerate(sites["station"]):
            long_rows.append({
                "scenario_id": args.scenario_id, "time": str(timestamp), "time_h": float(times_h[ti]),
                "station": str(station), "station_x_m": float(sites["x"].iloc[si]),
                "station_y_m": float(sites["y"].iloc[si]), "signal": float(signal[ti, si]),
                "background": float(background[ti, si]), "noiseless": float(noiseless[ti, si]),
                "observed": float(observed[ti, si]),
            })
    truth_path = output_dir / "truth_station_concentrations.csv"
    pd.DataFrame(long_rows).to_csv(truth_path, index=False, encoding="utf-8-sig")
    sites_copy = output_dir / f"sites{args.sites.suffix.lower()}"
    wind_copy = output_dir / f"wind{args.wind.suffix.lower()}"
    shutil.copy2(args.sites, sites_copy)
    shutil.copy2(args.wind, wind_copy)
    manifest = {
        "schema_version": 1, "scenario_id": args.scenario_id, "split": "development_extension",
        "truth_model": "lagrangian_particle_receptor_stress_model", "random_seed": args.seed,
        "source": {"x_m": source_x, "y_m": source_y, "lon": float(source_lon), "lat": float(source_lat),
                   "fraction_x": args.source_fraction_x, "fraction_y": args.source_fraction_y},
        "source_strength": {"shape": args.q_shape, "amplitude": amplitude,
                            "values_at_observation_times": [float(amplitude * value) for value in q_unit(times_h)],
                            "units": "arbitrary concentration-area per hour"},
        "physics": {"diffusivity_m2s": args.diffusivity_m2s, "initial_sigma_m": args.initial_sigma_m,
                    "decay_per_hour": args.decay_per_hour, "truth_wind_factor": args.truth_wind_factor,
                    "pre_event_hours": args.pre_event_hours, "integration_dt_h": None,
                    "particle_release_dt_h": args.release_dt_h, "particles_per_release": args.particles_per_release,
                    "diffusivity_along_m2s": along_d, "diffusivity_cross_m2s": cross_d,
                    "initial_sigma_along_m": along_sigma, "initial_sigma_cross_m": cross_sigma,
                    "sensor_kernel_sigma_m": args.sensor_kernel_sigma_m,
                    "meander_amplitude_m": args.meander_amplitude_m, "meander_period_h": args.meander_period_h},
        "observation_model": {"background_level": args.background_level,
                              "station_bias_fraction": args.station_bias_fraction,
                              "noise_absolute": args.noise_absolute, "noise_relative": args.noise_relative,
                              "realized_signal_peak": float(signal.max()), "realized_observed_peak": float(observed.max())},
        "model_scope": "transparent 2-D particle receptor stress generator; not FLEXPART or a validated operational LPDM",
        "files": {},
    }
    for label, path in {"sites": sites_copy, "wind": wind_copy, "concentration": concentration_path,
                        "truth_station_concentrations": truth_path}.items():
        manifest["files"][label] = {"path": str(path), "sha256": _sha256(path)}
    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scenario={args.scenario_id}")
    print(f"manifest_sha256={_sha256(manifest_path)}")


if __name__ == "__main__":
    main()
