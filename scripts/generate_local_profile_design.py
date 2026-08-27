"""Generate a truth-blind local fixed-source profile design from multistart runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


DIRECTIONS = (
    ("e", 1.0, 0.0),
    ("ne", 2**-0.5, 2**-0.5),
    ("n", 0.0, 1.0),
    ("nw", -(2**-0.5), 2**-0.5),
    ("w", -1.0, 0.0),
    ("sw", -(2**-0.5), -(2**-0.5)),
    ("s", 0.0, -1.0),
    ("se", 2**-0.5, -(2**-0.5)),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _unique_candidates(candidates: list[dict], tolerance_m: float = 1.0) -> list[dict]:
    unique = []
    for candidate in candidates:
        if any(
            math.hypot(
                candidate["x_m"] - previous["x_m"],
                candidate["y_m"] - previous["y_m"],
            )
            <= tolerance_m
            for previous in unique
        ):
            continue
        unique.append(candidate)
    return unique


def generate_design(
    multistart_root: Path,
    radii_m: list[float],
    nuisance_seed_offsets: list[int],
    include_basin_endpoints: bool = True,
) -> dict:
    """Build candidates without opening a truth manifest or truth summary."""
    plan_path = multistart_root / "experiment_plan.json"
    plan = _read_json(plan_path)
    bounds = plan["source_domain_m"]
    manifests = sorted(multistart_root.rglob("run_manifest.json"))
    if not manifests:
        raise ValueError(f"no run manifests found below {multistart_root}")

    runs = []
    for path in manifests:
        payload = _read_json(path)
        runtime = payload["runtime"]
        if runtime.get("source_position_mode") != "single":
            raise ValueError(f"non-single source run found: {path}")
        source = payload["source"]
        checkpoint = payload["checkpoint"]
        checkpoint_path = Path(checkpoint["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"run checkpoint does not exist: {checkpoint_path}")
        runs.append(
            {
                "start_id": str(runtime["run_id"]),
                "x_m": float(source["x_m"]),
                "y_m": float(source["y_m"]),
                "best_raw_loss": float(checkpoint["best_raw_loss"]),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "manifest_path": str(path.resolve()),
                "manifest_sha256": _sha256(path),
            }
        )
    selected = min(runs, key=lambda row: (row["best_raw_loss"], row["start_id"]))

    candidates = [
        {
            "candidate_id": "selected",
            "origin": f"selected_multistart::{selected['start_id']}",
            "x_m": selected["x_m"],
            "y_m": selected["y_m"],
        }
    ]
    for radius in radii_m:
        if radius <= 0:
            raise ValueError("profile radii must be positive")
        radius_id = f"r{int(round(radius)):04d}"
        for direction_id, dx, dy in DIRECTIONS:
            candidates.append(
                {
                    "candidate_id": f"{radius_id}_{direction_id}",
                    "origin": "selected_centered_radial_profile",
                    "x_m": _clip(
                        selected["x_m"] + radius * dx,
                        float(bounds["x_min_m"]),
                        float(bounds["x_max_m"]),
                    ),
                    "y_m": _clip(
                        selected["y_m"] + radius * dy,
                        float(bounds["y_min_m"]),
                        float(bounds["y_max_m"]),
                    ),
                }
            )
    if include_basin_endpoints:
        for run in sorted(runs, key=lambda row: row["start_id"]):
            candidates.append(
                {
                    "candidate_id": f"basin_{run['start_id']}",
                    "origin": f"multistart_endpoint::{run['start_id']}",
                    "x_m": run["x_m"],
                    "y_m": run["y_m"],
                }
            )
    candidates = _unique_candidates(candidates)

    width = float(bounds["x_max_m"]) - float(bounds["x_min_m"])
    height = float(bounds["y_max_m"]) - float(bounds["y_min_m"])
    starts = []
    for candidate in candidates:
        for seed_offset in nuisance_seed_offsets:
            if seed_offset < 0:
                raise ValueError("nuisance seed offsets must be non-negative")
            starts.append(
                {
                    "id": f"{candidate['candidate_id']}__n{seed_offset}",
                    "candidate_id": candidate["candidate_id"],
                    "nuisance_start_id": f"seed_offset_{seed_offset}",
                    "seed_offset": int(seed_offset),
                    "mode": "domain_fraction",
                    "fraction_x": (candidate["x_m"] - float(bounds["x_min_m"])) / width,
                    "fraction_y": (candidate["y_m"] - float(bounds["y_min_m"])) / height,
                }
            )

    return {
        "schema_version": 1,
        "design_id": f"local_profile_{multistart_root.name}_v1",
        "purpose": "Development-only truth-blind objective-consistent local source profile",
        "selection_rule": "minimum original best_raw_loss across nuisance starts at each fixed candidate",
        "truth_isolation": "generator reads only multistart experiment_plan.json and run_manifest.json files; it does not read scenario truth",
        "profile_geometry": {
            "center_rule": "minimum best_raw_loss multistart endpoint",
            "radii_m": [float(value) for value in radii_m],
            "directions": [row[0] for row in DIRECTIONS],
            "include_multistart_endpoints": bool(include_basin_endpoints),
            "deduplication_tolerance_m": 1.0,
        },
        "provenance": {
            "multistart_root": str(multistart_root.resolve()),
            "experiment_plan_path": str(plan_path.resolve()),
            "experiment_plan_sha256": _sha256(plan_path),
            "selected_start_id": selected["start_id"],
            "selected_best_raw_loss": selected["best_raw_loss"],
            "selected_checkpoint_path": selected["checkpoint_path"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "input_run_manifests": runs,
        },
        "resolved_candidates_m": candidates,
        "starts": starts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistart-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radii-m", nargs="+", type=float, default=[250.0, 500.0])
    parser.add_argument("--nuisance-seed-offsets", nargs="+", type=int, default=[0])
    parser.add_argument("--exclude-basin-endpoints", action="store_true")
    args = parser.parse_args()
    payload = generate_design(
        args.multistart_root,
        args.radii_m,
        args.nuisance_seed_offsets,
        include_basin_endpoints=not args.exclude_basin_endpoints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output.resolve()}")
    print(f"sha256={_sha256(args.output)}")
    print(f"candidate_count={len(payload['resolved_candidates_m'])}")
    print(f"run_count={len(payload['starts'])}")


if __name__ == "__main__":
    main()
