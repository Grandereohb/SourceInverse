"""Create a unified selected/oracle failure table across inversion methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _median(values):
    ordered = sorted(float(value) for value in values)
    return 0.5 * (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2])


def compare(
    baseline_csv: Path,
    gaussian_json: Path,
    pinn_csv: Path,
    pinn_json: Path,
) -> dict:
    with baseline_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    gaussian = json.loads(gaussian_json.read_text(encoding="utf-8"))
    with pinn_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        pinn_rows = list(csv.DictReader(handle))
    pinn = json.loads(pinn_json.read_text(encoding="utf-8"))

    rows = []
    for method in (
        "B0_maximum_anomaly_station",
        "B1_maximum_anomaly_station_fixed_upwind_shift",
    ):
        subset = [row for row in baseline_rows if row["method"] == method]
        errors = [float(row["localization_error_m"]) for row in subset]
        rows.append(
            {
                "method": method,
                "method_family": "geometric_heuristic",
                "relation_to_truth_model": "not_a_transport_model",
                "scenario_count": len(subset),
                "selected_success_count": sum(value <= 500.0 for value in errors),
                "selected_success_rate": sum(value <= 500.0 for value in errors)
                / len(errors),
                "selected_error_mean_m": sum(errors) / len(errors),
                "selected_error_median_m": _median(errors),
                "selected_error_max_m": max(errors),
                "oracle_success_count": "",
                "oracle_error_median_m": "",
                "selection_failure_reachable_count": "",
                "candidate_reachability_failure_count": "",
                "median_selection_regret_m": "",
                "total_wall_time_s": sum(
                    float(row["wall_time_s"] or 0.0) for row in subset
                ),
                "budget_description": "one deterministic estimate per event",
            }
        )
    for method, summary in gaussian["method_summaries"].items():
        rows.append(
            {
                "method": method,
                "method_family": "gaussian_puff_profile_grid",
                "relation_to_truth_model": "same forward-model family with fixed mismatched development physics",
                "scenario_count": summary["scenario_count"],
                "selected_success_count": summary["selected_success_count"],
                "selected_success_rate": summary["selected_success_count"]
                / summary["scenario_count"],
                "selected_error_mean_m": summary["selected_error_mean_m"],
                "selected_error_median_m": summary["selected_error_median_m"],
                "selected_error_max_m": max(
                    row["selected_localization_error_m"]
                    for row in gaussian["rows"]
                    if row["method"] == method
                ),
                "oracle_success_count": summary["oracle_success_count"],
                "oracle_error_median_m": summary["oracle_error_median_m"],
                "selection_failure_reachable_count": summary[
                    "failure_categories"
                ]["selection_failure_reachable"],
                "candidate_reachability_failure_count": summary[
                    "failure_categories"
                ]["candidate_reachability_failure"],
                "median_selection_regret_m": summary["median_selection_regret_m"],
                "total_wall_time_s": summary["total_wall_time_s"],
                "budget_description": f"{summary['total_forward_evaluation_count']} forward evaluations over 12 events",
            }
        )
    aggregate = pinn["aggregate"]
    rows.append(
        {
            "method": "differentiable_recurrent_transport_smooth_Q_6start",
            "method_family": "differentiable_recurrent_transport",
            "relation_to_truth_model": "different recurrent transport structure with fixed development physics",
            "scenario_count": aggregate["scenario_count"],
            "selected_success_count": aggregate["selected_success_count"],
            "selected_success_rate": aggregate["selected_success_rate"],
            "selected_error_mean_m": aggregate["selected_localization_error_m"]["mean"],
            "selected_error_median_m": aggregate["selected_localization_error_m"]["median"],
            "selected_error_max_m": aggregate["selected_localization_error_m"]["maximum"],
            "oracle_success_count": aggregate["oracle_success_count"],
            "oracle_error_median_m": aggregate["oracle_localization_error_m"]["median"],
            "selection_failure_reachable_count": aggregate[
                "failure_category_counts"
            ]["selection_failure_reachable"],
            "candidate_reachability_failure_count": aggregate[
                "failure_category_counts"
            ]["reachability_failure"],
            "median_selection_regret_m": aggregate["selection_regret_m"]["median"],
            "total_wall_time_s": aggregate["total_training_wall_time_s"],
            "budget_description": "72 gradient runs (6 starts x 12 events), 900 epochs each",
        }
    )

    pinn_by_id = {row["scenario_id"]: row for row in pinn_rows}
    paired = {}
    for gaussian_method in gaussian["method_summaries"]:
        gaussian_by_id = {
            row["scenario_id"]: row
            for row in gaussian["rows"]
            if row["method"] == gaussian_method
        }
        differences = [
            float(gaussian_by_id[scenario_id]["selected_localization_error_m"])
            - float(pinn_row["selected_localization_error_m"])
            for scenario_id, pinn_row in pinn_by_id.items()
        ]
        paired[gaussian_method] = {
            "scenario_count": len(differences),
            "gaussian_lower_error_count": sum(value < 0.0 for value in differences),
            "pinn_lower_error_count": sum(value > 0.0 for value in differences),
            "tie_count": sum(value == 0.0 for value in differences),
            "median_gaussian_minus_pinn_error_m": _median(differences),
            "mean_gaussian_minus_pinn_error_m": sum(differences) / len(differences),
        }
    return {
        "schema_version": 1,
        "analysis": "cross_method_failure_mechanism_comparison",
        "success_threshold_m": 500.0,
        "source_hashes": {
            "baseline_csv": _sha256(baseline_csv),
            "gaussian_json": _sha256(gaussian_json),
            "pinn_csv": _sha256(pinn_csv),
            "pinn_json": _sha256(pinn_json),
        },
        "method_rows": rows,
        "paired_gaussian_vs_pinn": paired,
        "caveat": "Gaussian puff shares the synthetic truth model family but uses fixed mismatched development physics; the recurrent method differs in both transport structure and optimization budget.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--gaussian-json", required=True, type=Path)
    parser.add_argument("--pinn-csv", required=True, type=Path)
    parser.add_argument("--pinn-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    payload = compare(
        args.baseline_csv, args.gaussian_json, args.pinn_csv, args.pinn_json
    )
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["method_rows"][0]))
        writer.writeheader()
        writer.writerows(payload["method_rows"])
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(json.dumps(payload["paired_gaussian_vs_pinn"], indent=2))


if __name__ == "__main__":
    main()
