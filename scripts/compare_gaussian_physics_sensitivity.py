"""Compare fixed-development and truth-coefficient Gaussian candidate surfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return 0.5 * (
        ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]
    )


def compare(fixed_path: Path, matched_path: Path) -> dict:
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    fixed_rows = {
        (row["scenario_id"], row["method"]): row for row in fixed["rows"]
    }
    matched_rows = {
        (row["scenario_id"], row["method"]): row for row in matched["rows"]
    }
    if fixed_rows.keys() != matched_rows.keys():
        raise ValueError("fixed and matched analyses do not contain identical cases")

    rows = []
    for key in sorted(fixed_rows):
        before = fixed_rows[key]
        after = matched_rows[key]
        fixed_error = float(before["selected_localization_error_m"])
        matched_error = float(after["selected_localization_error_m"])
        rows.append(
            {
                "scenario_id": key[0],
                "method": key[1],
                "fixed_selected_error_m": fixed_error,
                "coefficient_matched_selected_error_m": matched_error,
                "error_change_matched_minus_fixed_m": matched_error - fixed_error,
                "fixed_selected_success": bool(before["selected_success"]),
                "coefficient_matched_selected_success": bool(
                    after["selected_success"]
                ),
                "fixed_failure_category": before["failure_category"],
                "coefficient_matched_failure_category": after["failure_category"],
                "failure_transition": (
                    f"{before['failure_category']} -> {after['failure_category']}"
                ),
                "fixed_selected_anomaly_rmse": float(
                    before["selected_anomaly_rmse"]
                ),
                "coefficient_matched_selected_anomaly_rmse": float(
                    after["selected_anomaly_rmse"]
                ),
                "fixed_selection_regret_m": float(before["selection_regret_m"]),
                "coefficient_matched_selection_regret_m": float(
                    after["selection_regret_m"]
                ),
            }
        )

    summaries = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        differences = [
            row["error_change_matched_minus_fixed_m"] for row in subset
        ]
        transitions = {}
        for row in subset:
            transition = row["failure_transition"]
            transitions[transition] = transitions.get(transition, 0) + 1
        fixed_summary = fixed["method_summaries"][method]
        matched_summary = matched["method_summaries"][method]
        summaries.append(
            {
                "method": method,
                "scenario_count": len(subset),
                "fixed_selected_success_count": fixed_summary[
                    "selected_success_count"
                ],
                "coefficient_matched_selected_success_count": matched_summary[
                    "selected_success_count"
                ],
                "fixed_selection_failure_count": fixed_summary[
                    "failure_categories"
                ]["selection_failure_reachable"],
                "coefficient_matched_selection_failure_count": matched_summary[
                    "failure_categories"
                ]["selection_failure_reachable"],
                "fixed_selected_error_median_m": fixed_summary[
                    "selected_error_median_m"
                ],
                "coefficient_matched_selected_error_median_m": matched_summary[
                    "selected_error_median_m"
                ],
                "median_error_change_matched_minus_fixed_m": _median(differences),
                "mean_error_change_matched_minus_fixed_m": sum(differences)
                / len(differences),
                "improved_scenario_count": sum(value < 0.0 for value in differences),
                "worsened_scenario_count": sum(value > 0.0 for value in differences),
                "failure_transitions_json": json.dumps(
                    transitions, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return {
        "schema_version": 1,
        "analysis": "gaussian_fixed_vs_truth_coefficient_sensitivity",
        "source_hashes": {
            "fixed_truth_evaluation": _sha256(fixed_path),
            "coefficient_matched_truth_evaluation": _sha256(matched_path),
        },
        "scope": (
            "Truth-informed sensitivity only: diffusivity, initial sigma, decay, "
            "and wind factor are matched. Source coordinates remain hidden from "
            "the search. The inversion integration step remains 0.05 h while the "
            "truth generator used 0.02 h."
        ),
        "summary_rows": summaries,
        "scenario_rows": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-json", required=True, type=Path)
    parser.add_argument("--coefficient-matched-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-summary-csv", required=True, type=Path)
    parser.add_argument("--output-scenarios-csv", required=True, type=Path)
    args = parser.parse_args()
    payload = compare(args.fixed_json, args.coefficient_matched_json)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_summary_csv, payload["summary_rows"])
    _write_csv(args.output_scenarios_csv, payload["scenario_rows"])
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"summary_csv_sha256={_sha256(args.output_summary_csv)}")
    print(f"scenarios_csv_sha256={_sha256(args.output_scenarios_csv)}")
    print(json.dumps(payload["summary_rows"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
