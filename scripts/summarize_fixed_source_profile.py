"""Summarize nuisance-reoptimized fixed-source runs without probability claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _discover(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("run_manifest.json"))


def _resolved_start_metadata(root: Path) -> dict[str, dict]:
    plan_path = root / "experiment_plan.json"
    if not plan_path.is_file():
        return {}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mapping = {}
    for row in plan.get("resolved_starts", []):
        run_id = str(row.get("id", "")).strip()
        if run_id:
            mapping[run_id] = row
    return mapping


def summarize_profile(root: Path, fixed_tolerance_m: float = 1e-3) -> dict:
    runs = []
    start_metadata = _resolved_start_metadata(root)
    for manifest_path in _discover(root):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = manifest["runtime"]
        mode = runtime.get("source_position_mode", "single")
        if mode != "fixed":
            raise ValueError(f"non-fixed run found in profile root: {manifest_path}")
        initial = runtime.get("source_init_override_m")
        if not isinstance(initial, list) or len(initial) != 2:
            raise ValueError(f"fixed run lacks explicit source coordinates: {manifest_path}")
        final = [float(manifest["source"]["x_m"]), float(manifest["source"]["y_m"])]
        displacement = float(np.hypot(final[0] - initial[0], final[1] - initial[1]))
        if displacement > fixed_tolerance_m:
            raise ValueError(
                f"fixed source moved {displacement:.6g} m in {manifest_path}"
            )
        run_id = str(runtime.get("run_id") or "unlabelled")
        metadata = start_metadata.get(run_id, {})
        runs.append(
            {
                "solver": str(runtime["recurrent_solver"]),
                "run_id": run_id,
                "candidate_id": str(metadata.get("candidate_id", run_id)),
                "nuisance_start_id": str(
                    metadata.get("nuisance_start_id", "base")
                ),
                "random_seed": int(manifest.get("random_seed", 0)),
                "x_m": final[0],
                "y_m": final[1],
                "best_raw_loss": float(manifest["checkpoint"]["best_raw_loss"]),
                "best_epoch": int(manifest["checkpoint"]["best_epoch"]),
                "completed_epochs": int(runtime["completed_epochs"]),
                "training_wall_time_s": float(runtime["training_wall_time_s"]),
                "q_mode": str(runtime["q_mode"]),
                "manifest_sha256": _sha256(manifest_path),
                "run_dir": str(manifest_path.parent),
            }
        )
    if not runs:
        raise ValueError("No run_manifest.json files found.")
    keys = [(row["solver"], row["run_id"]) for row in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate solver/run combinations found.")
    grouped = {}
    for row in runs:
        grouped.setdefault(row["solver"], []).append(row)
    summaries = {}
    candidate_rows = []
    for solver, solver_rows in sorted(grouped.items()):
        candidate_groups = {}
        for row in solver_rows:
            candidate_groups.setdefault(row["candidate_id"], []).append(row)
        selected = []
        for candidate_id, nuisance_runs in sorted(candidate_groups.items()):
            x_values = [row["x_m"] for row in nuisance_runs]
            y_values = [row["y_m"] for row in nuisance_runs]
            coordinate_spread = float(
                np.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values))
            )
            if coordinate_spread > fixed_tolerance_m:
                raise ValueError(
                    f"candidate {candidate_id} maps to inconsistent coordinates"
                )
            best_nuisance = min(nuisance_runs, key=lambda row: row["best_raw_loss"])
            candidate = dict(best_nuisance)
            candidate["nuisance_run_count"] = len(nuisance_runs)
            selected.append(candidate)
        best_loss = min(row["best_raw_loss"] for row in selected)
        for row in selected:
            row["delta_profile_loss"] = row["best_raw_loss"] - best_loss
        candidate_rows.extend(selected)
        best = min(selected, key=lambda row: row["best_raw_loss"])
        summaries[solver] = {
            "candidate_count": len(selected),
            "nuisance_run_count": len(solver_rows),
            "minimum_nuisance_runs_per_candidate": min(
                row["nuisance_run_count"] for row in selected
            ),
            "maximum_nuisance_runs_per_candidate": max(
                row["nuisance_run_count"] for row in selected
            ),
            "best_candidate_id": best["candidate_id"],
            "best_nuisance_start_id": best["nuisance_start_id"],
            "best_x_m": best["x_m"],
            "best_y_m": best["y_m"],
            "best_raw_loss": best_loss,
            "maximum_delta_profile_loss": max(
                row["delta_profile_loss"] for row in selected
            ),
            "training_wall_time_total_s": sum(
                row["training_wall_time_s"] for row in solver_rows
            ),
        }
    return {
        "schema_version": 1,
        "analysis": "objective_consistent_fixed_source_profile",
        "profile_root": str(root.resolve()),
        "fixed_source_tolerance_m": float(fixed_tolerance_m),
        "solver_summaries": summaries,
        "rows": sorted(
            candidate_rows, key=lambda row: (row["solver"], row["candidate_id"])
        ),
        "runs": sorted(runs, key=lambda row: (row["solver"], row["run_id"])),
        "interpretation": (
            "Each candidate value is the minimum original training objective across "
            "its declared nuisance-parameter starts at a fixed source coordinate. "
            "Delta profile loss is not a posterior probability or confidence level; "
            "region thresholds require independent known-source calibration."
        ),
    }


def write_profile_csv(path: Path, payload: dict) -> None:
    columns = [
        "solver",
        "candidate_id",
        "run_id",
        "nuisance_start_id",
        "random_seed",
        "nuisance_run_count",
        "x_m",
        "y_m",
        "best_raw_loss",
        "delta_profile_loss",
        "best_epoch",
        "completed_epochs",
        "training_wall_time_s",
        "q_mode",
        "manifest_sha256",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in payload["rows"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--fixed-tolerance-m", type=float, default=1e-3)
    args = parser.parse_args()
    payload = summarize_profile(args.profile_root, args.fixed_tolerance_m)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_profile_csv(args.output_csv, payload)
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(json.dumps(payload["solver_summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
