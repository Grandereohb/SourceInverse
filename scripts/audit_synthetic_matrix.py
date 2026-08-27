"""Aggregate per-scenario QA without dropping difficult synthetic cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_synthetic_scenario import audit  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    batch = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    rows = []
    reports = []
    for entry in batch["scenarios"]:
        manifest_path = Path(entry["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = audit(manifest_path)
        reports.append(report)
        rows.append(
            {
                "scenario_id": report["scenario_id"],
                "input_id": entry.get("input_id"),
                "position_id": entry.get("design_factors", {}).get("position_id"),
                "physics_condition_id": entry.get("design_factors", {}).get(
                    "physics_condition_id"
                ),
                "source_x_m": manifest["source"]["x_m"],
                "source_y_m": manifest["source"]["y_m"],
                "q_shape": manifest["source_strength"]["shape"],
                "diffusivity_m2s": manifest["physics"]["diffusivity_m2s"],
                "initial_sigma_m": manifest["physics"]["initial_sigma_m"],
                "decay_per_hour": manifest["physics"]["decay_per_hour"],
                "truth_wind_factor": manifest["physics"]["truth_wind_factor"],
                "global_signal_peak": report["global_signal_peak"],
                "responsive_station_count_peak_gt_5pct_global": report[
                    "responsive_station_count_peak_gt_5pct_global"
                ],
                "dominant_signal_energy_ratio": report[
                    "dominant_signal_energy_ratio"
                ],
                "noise_rmse": report["noise_rmse"],
                "all_file_hashes_match": report["all_file_hashes_match"],
                "scenario_manifest_sha256": report["manifest_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "analysis": "synthetic_matrix_quality_audit",
        "batch_manifest_path": str(args.batch_manifest.resolve()),
        "batch_manifest_sha256": _sha256(args.batch_manifest),
        "scenario_count": len(rows),
        "all_file_hashes_match": all(row["all_file_hashes_match"] for row in rows),
        "retention_rule": batch["design"]["retention_rule"],
        "scenario_reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(f"scenario_count={len(rows)}")
    print(f"all_file_hashes_match={payload['all_file_hashes_match']}")


if __name__ == "__main__":
    main()
