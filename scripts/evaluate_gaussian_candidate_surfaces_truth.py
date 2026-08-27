"""Post-hoc truth evaluation of frozen Gaussian-puff candidate surfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return 0.5 * (ordered[(n - 1) // 2] + ordered[n // 2])


def _unique_candidates(candidates: list[dict]) -> list[dict]:
    by_coordinate = {}
    for candidate in candidates:
        key = (
            round(float(candidate["source_x_m"]), 6),
            round(float(candidate["source_y_m"]), 6),
        )
        if key not in by_coordinate or float(candidate["anomaly_rmse"]) < float(
            by_coordinate[key]["anomaly_rmse"]
        ):
            by_coordinate[key] = candidate
    return list(by_coordinate.values())


def evaluate(
    status_path: Path,
    legacy_manifest_path: Path | None = None,
    threshold_m=500.0,
) -> dict:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    legacy_by_scenario = None
    if legacy_manifest_path is not None:
        legacy = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        legacy_by_scenario = {
            row["scenario_id"]: row for row in legacy["outputs"]
        }
    rows = []
    for entry in status["outputs"]:
        scenario_path = Path(entry["scenario_manifest_path"])
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        truth_x = float(scenario["source"]["x_m"])
        truth_y = float(scenario["source"]["y_m"])
        surface_path = Path(entry["surface_path"])
        surface = json.loads(surface_path.read_text(encoding="utf-8"))
        legacy_results = None
        if legacy_by_scenario is not None:
            legacy_path = Path(
                legacy_by_scenario[entry["scenario_id"]]["gaussian_results_path"]
            )
            legacy_results = {
                result["method"]: result
                for result in json.loads(legacy_path.read_text(encoding="utf-8"))[
                    "results"
                ]
            }
        for result in surface["results"]:
            method = result["method"]
            candidates = _unique_candidates(result["candidates"])
            selected_loss = float(result["anomaly_rmse"])
            candidate_min = min(float(row["anomaly_rmse"]) for row in candidates)
            if not math.isclose(selected_loss, candidate_min, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"saved result is not candidate minimum: {method}")
            if legacy_results is not None:
                previous = legacy_results[method]
                for key in ("source_x_m", "source_y_m", "anomaly_rmse"):
                    if not math.isclose(
                        float(result[key]),
                        float(previous[key]),
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    ):
                        raise ValueError(
                            f"candidate-saving rerun changed legacy optimum {entry['scenario_id']} {method} {key}"
                        )
            for candidate in candidates:
                candidate["localization_error_m"] = math.hypot(
                    float(candidate["source_x_m"]) - truth_x,
                    float(candidate["source_y_m"]) - truth_y,
                )
            oracle = min(candidates, key=lambda row: row["localization_error_m"])
            selected_error = math.hypot(
                float(result["source_x_m"]) - truth_x,
                float(result["source_y_m"]) - truth_y,
            )
            oracle_error = float(oracle["localization_error_m"])
            selected_success = selected_error <= threshold_m
            oracle_success = oracle_error <= threshold_m
            category = (
                "selected_success"
                if selected_success
                else (
                    "selection_failure_reachable"
                    if oracle_success
                    else "candidate_reachability_failure"
                )
            )
            losses = pd.Series([float(row["anomaly_rmse"]) for row in candidates])
            errors = pd.Series(
                [float(row["localization_error_m"]) for row in candidates]
            )
            spearman = float(losses.rank().corr(errors.rank()))
            within = [
                row
                for row in candidates
                if float(row["anomaly_rmse"]) <= selected_loss * 1.05
            ]
            if len(within) > 1:
                coordinates = np.array(
                    [[row["source_x_m"], row["source_y_m"]] for row in within],
                    dtype=float,
                )
                differences = coordinates[:, None, :] - coordinates[None, :, :]
                near_spread = float(np.sqrt(np.sum(differences**2, axis=2)).max())
            else:
                near_spread = 0.0
            rows.append(
                {
                    "scenario_id": entry["scenario_id"],
                    "input_id": entry.get("input_id"),
                    "position_id": entry.get("design_factors", {}).get(
                        "position_id"
                    ),
                    "physics_condition_id": entry.get("design_factors", {}).get(
                        "physics_condition_id"
                    ),
                    "method": method,
                    "candidate_count_raw": len(result["candidates"]),
                    "candidate_count_unique": len(candidates),
                    "selected_x_m": float(result["source_x_m"]),
                    "selected_y_m": float(result["source_y_m"]),
                    "selected_anomaly_rmse": selected_loss,
                    "selected_localization_error_m": selected_error,
                    "selected_success": selected_success,
                    "oracle_x_m": float(oracle["source_x_m"]),
                    "oracle_y_m": float(oracle["source_y_m"]),
                    "oracle_localization_error_m": oracle_error,
                    "oracle_success": oracle_success,
                    "selection_regret_m": selected_error - oracle_error,
                    "failure_category": category,
                    "near_5pct_candidate_count": len(within),
                    "near_5pct_max_spread_m": near_spread,
                    "spearman_loss_vs_localization_error": spearman,
                    "forward_evaluation_count": int(result["forward_evaluation_count"]),
                    "wall_time_s": float(result["wall_time_s"]),
                    "surface_sha256": _sha256(surface_path),
                    "scenario_manifest_sha256": _sha256(scenario_path),
                }
            )
    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        selected_errors = [row["selected_localization_error_m"] for row in subset]
        oracle_errors = [row["oracle_localization_error_m"] for row in subset]
        categories = {
            name: sum(row["failure_category"] == name for row in subset)
            for name in (
                "selected_success",
                "selection_failure_reachable",
                "candidate_reachability_failure",
            )
        }
        summaries[method] = {
            "scenario_count": len(subset),
            "selected_success_count": sum(row["selected_success"] for row in subset),
            "oracle_success_count": sum(row["oracle_success"] for row in subset),
            "selected_error_median_m": _median(selected_errors),
            "oracle_error_median_m": _median(oracle_errors),
            "selected_error_mean_m": sum(selected_errors) / len(selected_errors),
            "oracle_error_mean_m": sum(oracle_errors) / len(oracle_errors),
            "failure_categories": categories,
            "median_selection_regret_m": _median(
                [row["selection_regret_m"] for row in subset]
            ),
            "total_wall_time_s": sum(row["wall_time_s"] for row in subset),
            "total_forward_evaluation_count": sum(
                row["forward_evaluation_count"] for row in subset
            ),
        }
    return {
        "schema_version": 1,
        "analysis": "truth_known_gaussian_candidate_surface_evaluation",
        "truth_join_timing": "truth is read only after all candidate surfaces are frozen",
        "success_threshold_m": float(threshold_m),
        "status_path": str(status_path.resolve()),
        "status_sha256": _sha256(status_path),
        "legacy_manifest_path": (
            str(legacy_manifest_path.resolve())
            if legacy_manifest_path is not None
            else None
        ),
        "legacy_manifest_sha256": (
            _sha256(legacy_manifest_path)
            if legacy_manifest_path is not None
            else None
        ),
        "legacy_optimum_validation": (
            "all selected coordinates and losses exactly reproduced within 1e-10"
            if legacy_manifest_path is not None
            else "not_requested"
        ),
        "method_summaries": summaries,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--legacy-baseline-manifest", type=Path)
    parser.add_argument("--threshold-m", type=float, default=500.0)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    payload = evaluate(args.status, args.legacy_baseline_manifest, args.threshold_m)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0]))
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(json.dumps(payload["method_summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
