"""Post-hoc truth evaluation for a frozen geometric-baseline matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _median(values: list[float]) -> float:
    values = sorted(values)
    return 0.5 * (values[(len(values) - 1) // 2] + values[len(values) // 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    status = json.loads(args.status.read_text(encoding="utf-8"))
    rows = []
    for entry in status["outputs"]:
        scenario = json.loads(Path(entry["scenario_manifest_path"]).read_text(encoding="utf-8"))
        truth_x = float(scenario["source"]["x_m"])
        truth_y = float(scenario["source"]["y_m"])
        results = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))["results"]
        for result in results:
            error = math.hypot(float(result["source_x_m"]) - truth_x, float(result["source_y_m"]) - truth_y)
            rows.append(
                {
                    "scenario_id": entry["scenario_id"],
                    "input_id": entry.get("input_id"),
                    "position_id": entry.get("design_factors", {}).get("position_id"),
                    "physics_condition_id": entry.get("design_factors", {}).get("physics_condition_id"),
                    "method": result["method"],
                    "localization_error_m": error,
                    "success_le_500m": error <= 500.0,
                    "estimate_x_m": result["source_x_m"],
                    "estimate_y_m": result["source_y_m"],
                }
            )
    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        errors = [row["localization_error_m"] for row in subset]
        summaries[method] = {
            "scenario_count": len(subset),
            "success_count": sum(row["success_le_500m"] for row in subset),
            "median_error_m": _median(errors),
            "mean_error_m": sum(errors) / len(errors),
            "maximum_error_m": max(errors),
        }
    payload = {
        "schema_version": 1,
        "analysis": "truth_known_geometric_baseline_matrix_evaluation",
        "truth_join_timing": "truth read only after all baseline estimates were frozen",
        "status_sha256": _sha256(args.status),
        "method_summaries": summaries,
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
