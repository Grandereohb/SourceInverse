"""Prepare a truth-blind dominant-station holdout input and one-start design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def prepare_holdout(
    concentration_path: Path,
    multistart_root: Path,
    profile_design_path: Path,
) -> tuple[pd.DataFrame, dict, dict]:
    frame = pd.read_csv(concentration_path)
    target_columns = {"TARGET_POLLUTANT"}
    station_columns = [
        column
        for column in frame.columns[1:]
        if str(column) not in target_columns
    ]
    if len(station_columns) < 3:
        raise ValueError("dominant-station holdout requires at least three stations")
    values = frame[station_columns].astype(float)
    baseline = values.median(axis=1).to_numpy(dtype=float)
    residual = np.clip(values.to_numpy(dtype=float) - baseline[:, None], 0.0, None)
    energies = np.sum(residual**2, axis=0)
    dominant_index = int(np.argmax(energies))
    dominant_station = station_columns[dominant_index]

    profile_design = json.loads(profile_design_path.read_text(encoding="utf-8"))
    selected_manifest_path = Path(
        next(
            row["manifest_path"]
            for row in profile_design["provenance"]["input_run_manifests"]
            if row["start_id"] == profile_design["provenance"]["selected_start_id"]
        )
    )
    selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    selected_source = selected_manifest["source"]
    plan_path = multistart_root / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bounds = plan["source_domain_m"]
    width = float(bounds["x_max_m"]) - float(bounds["x_min_m"])
    height = float(bounds["y_max_m"]) - float(bounds["y_min_m"])
    fraction_x = (float(selected_source["x_m"]) - float(bounds["x_min_m"])) / width
    fraction_y = (float(selected_source["y_m"]) - float(bounds["y_min_m"])) / height
    design = {
        "schema_version": 1,
        "design_id": f"dominant_station_holdout_{multistart_root.name}_v1",
        "purpose": "Truth-blind dominant-station deletion refit from selected checkpoint",
        "selection_rule": "single warm-start refit after deleting the largest positive-residual-energy station",
        "truth_isolation": "station selection uses concentrations only and does not read source truth",
        "starts": [
            {
                "id": "dominant_station_holdout",
                "mode": "domain_fraction",
                "fraction_x": fraction_x,
                "fraction_y": fraction_y,
            }
        ],
    }
    training_frame = frame.drop(columns=[dominant_station])
    total_energy = float(np.sum(energies))
    manifest = {
        "schema_version": 1,
        "analysis": "dominant_station_holdout_preparation",
        "concentration_path": str(concentration_path.resolve()),
        "concentration_sha256": _sha256(concentration_path),
        "multistart_root": str(multistart_root.resolve()),
        "experiment_plan_sha256": _sha256(plan_path),
        "profile_design_path": str(profile_design_path.resolve()),
        "profile_design_sha256": _sha256(profile_design_path),
        "selected_start_id": profile_design["provenance"]["selected_start_id"],
        "selected_manifest_path": str(selected_manifest_path.resolve()),
        "selected_manifest_sha256": _sha256(selected_manifest_path),
        "selected_checkpoint_path": profile_design["provenance"][
            "selected_checkpoint_path"
        ],
        "selected_checkpoint_sha256": profile_design["provenance"][
            "selected_checkpoint_sha256"
        ],
        "selected_source_x_m": float(selected_source["x_m"]),
        "selected_source_y_m": float(selected_source["y_m"]),
        "heldout_station": dominant_station,
        "heldout_station_positive_residual_energy": float(energies[dominant_index]),
        "heldout_station_energy_ratio": (
            float(energies[dominant_index]) / total_energy if total_energy else 0.0
        ),
        "station_count_before": len(station_columns),
        "station_count_after": len(station_columns) - 1,
    }
    return training_frame, design, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concentration", required=True, type=Path)
    parser.add_argument("--multistart-root", required=True, type=Path)
    parser.add_argument("--profile-design", required=True, type=Path)
    parser.add_argument("--output-concentration", required=True, type=Path)
    parser.add_argument("--output-design", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    frame, design, manifest = prepare_holdout(
        args.concentration, args.multistart_root, args.profile_design
    )
    for path in (args.output_concentration, args.output_design, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_concentration, index=False, encoding="utf-8-sig")
    args.output_design.write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["training_concentration_path"] = str(
        args.output_concentration.resolve()
    )
    manifest["training_concentration_sha256"] = _sha256(args.output_concentration)
    manifest["design_path"] = str(args.output_design.resolve())
    manifest["design_sha256"] = _sha256(args.output_design)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"heldout_station={manifest['heldout_station']}")
    print(f"energy_ratio={manifest['heldout_station_energy_ratio']:.8f}")
    print(f"manifest_sha256={_sha256(args.output_manifest)}")


if __name__ == "__main__":
    main()
