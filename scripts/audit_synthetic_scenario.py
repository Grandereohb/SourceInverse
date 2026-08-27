"""Create a machine-readable QA report for one generated synthetic scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_entry = manifest["files"]["truth_station_concentrations"]
    truth_path = Path(truth_entry["path"])
    truth = pd.read_csv(truth_path, encoding="utf-8-sig")
    required = {
        "scenario_id",
        "time",
        "station",
        "signal",
        "background",
        "noiseless",
        "observed",
    }
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"Truth table is missing columns: {missing}")
    if truth.empty:
        raise ValueError("Truth table is empty.")
    signal = truth["signal"].astype(float)
    noise = truth["observed"].astype(float) - truth["noiseless"].astype(float)
    station_energy = (
        truth.assign(signal_energy=signal**2)
        .groupby("station", sort=False)["signal_energy"]
        .sum()
        .sort_values(ascending=False)
    )
    station_peaks = (
        truth.groupby("station", sort=False)["signal"].max().sort_values(ascending=False)
    )
    total_energy = float(station_energy.sum())
    global_peak = float(signal.max())
    file_checks = {}
    for label, entry in manifest["files"].items():
        path = Path(entry["path"])
        actual = _sha256(path)
        file_checks[label] = {
            "path": str(path.resolve()),
            "expected_sha256": str(entry["sha256"]).upper(),
            "actual_sha256": actual,
            "matches": actual == str(entry["sha256"]).upper(),
        }
    return {
        "schema_version": 1,
        "analysis": "synthetic_scenario_quality_audit",
        "scenario_id": manifest["scenario_id"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "truth_model": manifest["truth_model"],
        "n_rows": int(len(truth)),
        "n_times": int(truth["time"].nunique()),
        "n_stations": int(truth["station"].nunique()),
        "global_signal_peak": global_peak,
        "responsive_station_count_peak_gt_1": int((station_peaks > 1.0).sum()),
        "responsive_station_count_peak_gt_5pct_global": int(
            (station_peaks > 0.05 * global_peak).sum()
        ),
        "dominant_signal_energy_ratio": (
            float(station_energy.iloc[0] / total_energy) if total_energy > 0 else None
        ),
        "station_signal_energy_shares": {
            str(station): float(value / total_energy)
            for station, value in station_energy.items()
        },
        "station_signal_peaks": {
            str(station): float(value) for station, value in station_peaks.items()
        },
        "noise_mean": float(noise.mean()),
        "noise_rmse": float(math.sqrt(float((noise**2).mean()))),
        "all_file_hashes_match": all(row["matches"] for row in file_checks.values()),
        "file_checks": file_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit(args.scenario_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output.resolve()}")
    print(f"sha256={_sha256(args.output)}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
