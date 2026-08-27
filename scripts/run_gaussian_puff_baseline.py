"""Run a truth-blind constant-Q Gaussian-puff profile-grid baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from data_io import load_conc, load_sites, load_wind, wind_dir_to_uv  # noqa: E402
from gaussian_puff_baseline import (  # noqa: E402
    search_constant_q_gaussian_puff,
    search_dynamic_q_gaussian_puff,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--concentration", required=True, type=Path)
    parser.add_argument("--wind", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-position-pad-m", type=float, default=500.0)
    parser.add_argument("--diffusivity-m2s", type=float, default=8.0)
    parser.add_argument("--initial-sigma-m", type=float, default=120.0)
    parser.add_argument("--decay-per-hour", type=float, default=0.08)
    parser.add_argument("--wind-factor", type=float, default=0.35)
    parser.add_argument("--pre-event-hours", type=float, default=1.0)
    parser.add_argument("--integration-dt-h", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--refinement-levels", type=int, default=3)
    parser.add_argument("--dynamic-q-node-count", type=int, default=5)
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument(
        "--parameter-source",
        choices=[
            "development_candidate",
            "oracle_truth_for_smoke",
            "truth_informed_matched_physics_sensitivity",
        ],
        default="development_candidate",
    )
    args = parser.parse_args()
    inputs = {
        "sites": args.sites.resolve(),
        "concentration": args.concentration.resolve(),
        "wind": args.wind.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    sites, _, _ = load_sites(inputs["sites"])
    concentration = load_conc(inputs["concentration"])
    wind = load_wind(inputs["wind"])
    merged = concentration.merge(wind, on="time", how="inner")
    station_names = [
        name
        for name in sites["station"].astype(str)
        if name in merged.columns and not merged[name].isna().all()
    ]
    merged = merged.dropna(subset=["dir", "sp"] + station_names).copy()
    site_rows = sites.set_index("station").loc[station_names]
    values = merged[station_names].astype(float)
    anomaly = values.subtract(values.median(axis=1), axis=0).clip(lower=0.0)
    times_h = (
        (merged["time"] - merged["time"].iloc[0]).dt.total_seconds().to_numpy()
        / 3600.0
    )
    wind_u, wind_v = wind_dir_to_uv(
        merged["dir"].to_numpy(dtype=np.float64),
        merged["sp"].to_numpy(dtype=np.float64),
        is_from=True,
    )
    pad = float(args.source_position_pad_m)
    bounds = (
        float(sites["x"].min() - pad),
        float(sites["x"].max() + pad),
        float(sites["y"].min() - pad),
        float(sites["y"].max() + pad),
    )
    start = time.perf_counter()
    shared = dict(
        station_x_m=site_rows["x"].to_numpy(),
        station_y_m=site_rows["y"].to_numpy(),
        observation_times_h=times_h,
        wind_times_h=times_h,
        wind_u_mps=wind_u,
        wind_v_mps=wind_v,
        target_anomaly=anomaly.to_numpy(dtype=np.float64),
        source_bounds_m=bounds,
        diffusivity_m2s=args.diffusivity_m2s,
        initial_sigma_m=args.initial_sigma_m,
        decay_per_hour=args.decay_per_hour,
        wind_factor=args.wind_factor,
        pre_event_hours=args.pre_event_hours,
        integration_dt_h=args.integration_dt_h,
        grid_size=args.grid_size,
        refinement_levels=args.refinement_levels,
        return_candidates=args.save_candidates,
    )
    constant_result = search_constant_q_gaussian_puff(**shared)
    constant_result["wall_time_s"] = float(time.perf_counter() - start)
    dynamic_start = time.perf_counter()
    dynamic_shared = dict(shared)
    dynamic_shared["q_node_count"] = args.dynamic_q_node_count
    dynamic_result = search_dynamic_q_gaussian_puff(**dynamic_shared)
    dynamic_result["wall_time_s"] = float(time.perf_counter() - dynamic_start)
    truth_informed = (
        args.parameter_source == "truth_informed_matched_physics_sensitivity"
    )
    payload = {
        "schema_version": 1,
        "analysis": (
            "truth_informed_matched_physics_gaussian_sensitivity"
            if truth_informed
            else "truth_blind_gaussian_puff_baseline"
        ),
        "truth_isolation": (
            "physics values are truth-informed; source coordinates are not provided"
            if truth_informed
            else "runner does not accept or read source truth"
        ),
        "development_notice": (
            "This is a non-deployable truth-informed sensitivity analysis."
            if truth_informed
            else (
                "Physics and search settings are development candidates until "
                "frozen before test evaluation."
            )
        ),
        "physics_parameter_source": args.parameter_source,
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "results": [constant_result, dynamic_result],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output.resolve()}")
    print(f"sha256={_sha256(args.output)}")
    summaries = []
    for result in payload["results"]:
        summary = {
            key: result[key]
            for key in (
                "method",
                "source_x_m",
                "source_y_m",
                "anomaly_rmse",
                "normalized_anomaly_rmse",
                "forward_evaluation_count",
                "wall_time_s",
            )
        }
        if "candidates" in result:
            summary["saved_candidate_count"] = len(result["candidates"])
        summaries.append(summary)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
