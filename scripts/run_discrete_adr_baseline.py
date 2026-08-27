"""Run a truth-blind conventional discrete upwind-ADR profile baseline."""

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
from discrete_adr_baseline import search_discrete_adr_profiles  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--concentration", required=True, type=Path)
    parser.add_argument("--wind", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-position-pad-m", type=float, default=500.0)
    parser.add_argument("--transport-pad-m", type=float, default=3500.0)
    parser.add_argument("--diffusivity-m2s", type=float, default=2.0)
    parser.add_argument("--initial-sigma-m", type=float, default=334.713)
    parser.add_argument("--decay-per-hour", type=float, default=0.5)
    parser.add_argument("--wind-factor", type=float, default=0.25)
    parser.add_argument("--pre-event-hours", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--refinement-levels", type=int, default=3)
    parser.add_argument("--dynamic-q-node-count", type=int, default=5)
    parser.add_argument("--transport-grid-nx", type=int, default=32)
    parser.add_argument("--transport-grid-ny", type=int, default=32)
    parser.add_argument("--save-candidates", action="store_true")
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
    source_pad = float(args.source_position_pad_m)
    transport_pad = float(args.transport_pad_m)
    source_bounds = (
        float(sites["x"].min() - source_pad),
        float(sites["x"].max() + source_pad),
        float(sites["y"].min() - source_pad),
        float(sites["y"].max() + source_pad),
    )
    transport_bounds = (
        float(sites["x"].min() - transport_pad),
        float(sites["x"].max() + transport_pad),
        float(sites["y"].min() - transport_pad),
        float(sites["y"].max() + transport_pad),
    )
    start = time.perf_counter()
    results = search_discrete_adr_profiles(
        station_x_m=site_rows["x"].to_numpy(),
        station_y_m=site_rows["y"].to_numpy(),
        observation_times_h=times_h,
        wind_times_h=times_h,
        wind_u_mps=wind_u,
        wind_v_mps=wind_v,
        target_anomaly=anomaly.to_numpy(dtype=np.float64),
        source_bounds_m=source_bounds,
        transport_bounds_m=transport_bounds,
        diffusivity_m2s=args.diffusivity_m2s,
        initial_sigma_m=args.initial_sigma_m,
        decay_per_hour=args.decay_per_hour,
        wind_factor=args.wind_factor,
        q_node_count=args.dynamic_q_node_count,
        grid_size=args.grid_size,
        refinement_levels=args.refinement_levels,
        transport_grid_nx=args.transport_grid_nx,
        transport_grid_ny=args.transport_grid_ny,
        pre_event_hours=args.pre_event_hours,
        return_candidates=args.save_candidates,
    )
    wall_time = float(time.perf_counter() - start)
    for result in results:
        result["wall_time_s"] = wall_time
    payload = {
        "schema_version": 1,
        "analysis": "truth_blind_conventional_discrete_ADR_candidate_surfaces",
        "truth_isolation": "runner does not accept or read source truth",
        "model_scope": (
            "first-order upwind finite-volume-like ADR baseline on a padded 2-D "
            "domain; not operational CFD or LPDM"
        ),
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output.resolve()}")
    print(f"sha256={_sha256(args.output)}")
    print(
        json.dumps(
            [
                {
                    key: result[key]
                    for key in (
                        "method",
                        "source_x_m",
                        "source_y_m",
                        "anomaly_rmse",
                        "normalized_anomaly_rmse",
                        "candidate_count",
                        "forward_evaluation_count",
                        "wall_time_s",
                    )
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
