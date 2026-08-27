"""Aggregate post-optimization multistart truth summaries across a frozen matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _failure_category(selected_error: float, oracle_error: float, threshold: float) -> str:
    if selected_error <= threshold:
        return "selected_success"
    if oracle_error <= threshold:
        return "selection_failure_reachable"
    return "reachability_failure"


def _distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize an empty distribution.")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty scenario set.")
    scenario_count = len(rows)
    category_counts = {
        category: sum(row["failure_category"] == category for row in rows)
        for category in (
            "selected_success",
            "selection_failure_reachable",
            "reachability_failure",
        )
    }
    selected_success_count = sum(row["selected_success"] for row in rows)
    oracle_success_count = sum(row["oracle_success"] for row in rows)
    truth_coincident_start_count = sum(row["truth_coincident_start"] for row in rows)
    return {
        "scenario_count": scenario_count,
        "selected_success_count": selected_success_count,
        "selected_success_rate": selected_success_count / scenario_count,
        "oracle_success_count": oracle_success_count,
        "oracle_success_rate": oracle_success_count / scenario_count,
        "truth_coincident_start_count": truth_coincident_start_count,
        "truth_coincident_start_rate": truth_coincident_start_count / scenario_count,
        "failure_category_counts": category_counts,
        "failure_category_rates": {
            key: value / scenario_count for key, value in category_counts.items()
        },
        "selected_localization_error_m": _distribution(
            [row["selected_localization_error_m"] for row in rows]
        ),
        "oracle_localization_error_m": _distribution(
            [row["oracle_localization_error_m"] for row in rows]
        ),
        "selection_regret_m": _distribution(
            [row["selection_regret_m"] for row in rows]
        ),
        "total_training_wall_time_s": sum(
            row["total_training_wall_time_s"] for row in rows
        ),
    }


def _scenario_row(
    expected: dict,
    summary_path: Path,
    solver: str,
    success_threshold_m: float,
) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_id = str(expected["scenario_id"])
    actual_id = str(payload["scenario"]["id"])
    if actual_id != expected_id:
        raise ValueError(
            f"Scenario mismatch for {summary_path}: expected {expected_id}, got {actual_id}."
        )
    if payload.get("truth_join_timing") != "truth is read only by this post-optimization summarizer":
        raise ValueError(f"Unexpected truth-join declaration in {summary_path}.")
    try:
        summary = payload["solver_summaries"][solver]
    except KeyError as exc:
        raise ValueError(f"Solver {solver!r} missing from {summary_path}.") from exc
    solver_runs = [row for row in payload["runs"] if str(row["solver"]) == solver]
    selected_start_id = str(summary["selected_start_id"])
    selected_run = next(
        (
            row
            for row in solver_runs
            if str(row["start_id"]) == selected_start_id
        ),
        None,
    )
    if selected_run is None:
        raise ValueError(f"Selected run {selected_start_id!r} missing from {summary_path}.")
    selected_quality_path = Path(selected_run["run_dir"]) / "result_quality_report.json"
    if not selected_quality_path.is_file():
        raise FileNotFoundError(
            f"Selected quality report missing for {actual_id}: {selected_quality_path}"
        )
    selected_quality_sha256 = _sha256(selected_quality_path)
    if selected_quality_sha256 != str(selected_run["quality_report_sha256"]):
        raise ValueError(f"Selected quality report hash mismatch for {actual_id}.")
    selected_quality = json.loads(selected_quality_path.read_text(encoding="utf-8"))
    source_quality = selected_quality["source"]
    recurrent_quality = selected_quality["recurrent_pde"]
    dominant_quality = selected_quality["quality_diagnostics"]["dominant_station"]
    oracle_start_id = str(summary["oracle_start_id"])
    if not any(str(row["start_id"]) == oracle_start_id for row in solver_runs):
        raise ValueError(f"Oracle run {oracle_start_id!r} missing from {summary_path}.")
    closest_initial_run = min(
        solver_runs,
        key=lambda row: _finite(row["initial_error_m"], "initial error"),
    )
    minimum_initial_error = _finite(
        closest_initial_run["initial_error_m"], "minimum initial error"
    )
    selected_error = _finite(
        summary["selected_localization_error_m"], "selected localization error"
    )
    oracle_error = _finite(
        summary["oracle_localization_error_m"], "oracle localization error"
    )
    regret = _finite(summary["selection_regret_m"], "selection regret")
    if selected_error + 1e-8 < oracle_error:
        raise ValueError(f"Selected error is below oracle error in {summary_path}.")
    if not math.isclose(regret, selected_error - oracle_error, abs_tol=1e-8):
        raise ValueError(f"Selection regret is inconsistent in {summary_path}.")
    return {
        "scenario_id": actual_id,
        "closest_initial_start_id": str(closest_initial_run["start_id"]),
        "minimum_initial_error_m": minimum_initial_error,
        "truth_coincident_start": minimum_initial_error <= 1e-6,
        "selected_start_id": selected_start_id,
        "selected_initial_error_m": _finite(
            selected_run["initial_error_m"], "selected initial error"
        ),
        "selected_localization_error_m": selected_error,
        "selected_success": selected_error <= success_threshold_m,
        "oracle_start_id": oracle_start_id,
        "oracle_localization_error_m": oracle_error,
        "oracle_success": oracle_error <= success_threshold_m,
        "selection_regret_m": regret,
        "failure_category": _failure_category(
            selected_error, oracle_error, success_threshold_m
        ),
        "selected_best_raw_loss": _finite(
            summary["selected_best_raw_loss"], "selected best raw loss"
        ),
        "selected_fit_raw_rmse": _finite(
            selected_run["fit_raw_rmse"], "selected fit raw RMSE"
        ),
        "selected_optimizer_displacement_m": _finite(
            selected_run["optimizer_displacement_m"],
            "selected optimizer displacement",
        ),
        "selected_warning_count": int(selected_run["warning_count"]),
        "selected_min_boundary_margin_m": _finite(
            source_quality["min_boundary_margin_m"], "selected boundary margin"
        ),
        "selected_dominant_station_ratio": _finite(
            dominant_quality["residual_energy_ratio"],
            "selected dominant-station ratio",
        ),
        "selected_substep_cap_hit_count": int(
            recurrent_quality["substep_cap_hit_count"]
        ),
        "selected_source_quadrature_cap_hit_count": int(
            recurrent_quality.get("source_quadrature_cap_hit_count", 0)
        ),
        "selected_is_reasonable": bool(selected_quality["is_reasonable"]),
        "second_best_loss_relative_gap": (
            None
            if summary["second_best_loss_relative_gap"] is None
            else _finite(
                summary["second_best_loss_relative_gap"],
                "second-best relative loss gap",
            )
        ),
        "near_optimal_count": int(summary["near_optimal_count"]),
        "near_optimal_max_source_spread_m": _finite(
            summary["near_optimal_max_source_spread_m"],
            "near-optimal source spread",
        ),
        "spearman_loss_vs_localization_error": (
            None
            if summary["spearman_loss_vs_localization_error"] is None
            else _finite(
                summary["spearman_loss_vs_localization_error"],
                "loss-error Spearman correlation",
            )
        ),
        "total_training_wall_time_s": _finite(
            summary["total_training_wall_time_s"], "training wall time"
        ),
        "truth_summary_path": str(summary_path.resolve()),
        "truth_summary_sha256": _sha256(summary_path),
    }


def aggregate(
    matrix_plan_path: Path,
    analysis_root: Path,
    *,
    solver: str = "production",
    success_threshold_m: float = 500.0,
    summary_filename: str = "truth_known_multistart_summary.json",
) -> dict:
    if not math.isfinite(success_threshold_m) or success_threshold_m <= 0.0:
        raise ValueError("success_threshold_m must be finite and positive.")
    matrix_plan = json.loads(matrix_plan_path.read_text(encoding="utf-8"))
    expected_scenarios = list(matrix_plan["scenarios"])
    if len(expected_scenarios) != int(matrix_plan["scenario_count"]):
        raise ValueError("Matrix plan scenario_count does not match its scenario list.")
    expected_ids = [str(row["scenario_id"]) for row in expected_scenarios]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Matrix plan contains duplicate scenario IDs.")
    rows = []
    for expected in expected_scenarios:
        summary_path = (
            analysis_root / str(expected["scenario_id"]) / summary_filename
        )
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing truth summary for {expected['scenario_id']}: {summary_path}"
            )
        rows.append(
            _scenario_row(expected, summary_path, solver, success_threshold_m)
        )
    scenario_count = len(rows)
    noncoincident_rows = [row for row in rows if not row["truth_coincident_start"]]
    return {
        "schema_version": 1,
        "analysis": "truth_known_multistart_matrix_posthoc_evaluation",
        "truth_join_timing": "truth summaries are read only after every inversion run is complete",
        "matrix_plan": {
            "path": str(matrix_plan_path.resolve()),
            "sha256": _sha256(matrix_plan_path),
        },
        "analysis_root": str(analysis_root.resolve()),
        "solver": solver,
        "success_threshold_m": success_threshold_m,
        "scenario_count": scenario_count,
        "aggregate": _aggregate_metrics(rows),
        "sensitivity_excluding_truth_coincident_starts": (
            _aggregate_metrics(noncoincident_rows) if noncoincident_rows else None
        ),
        "scenarios": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scenario_id",
        "closest_initial_start_id",
        "minimum_initial_error_m",
        "truth_coincident_start",
        "selected_start_id",
        "selected_initial_error_m",
        "selected_localization_error_m",
        "selected_success",
        "oracle_start_id",
        "oracle_localization_error_m",
        "oracle_success",
        "selection_regret_m",
        "failure_category",
        "selected_best_raw_loss",
        "selected_fit_raw_rmse",
        "selected_optimizer_displacement_m",
        "selected_warning_count",
        "selected_min_boundary_margin_m",
        "selected_dominant_station_ratio",
        "selected_substep_cap_hit_count",
        "selected_source_quadrature_cap_hit_count",
        "selected_is_reasonable",
        "second_best_loss_relative_gap",
        "near_optimal_count",
        "near_optimal_max_source_spread_m",
        "spearman_loss_vs_localization_error",
        "total_training_wall_time_s",
        "truth_summary_path",
        "truth_summary_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-plan", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--solver", default="production")
    parser.add_argument("--success-threshold-m", type=float, default=500.0)
    parser.add_argument(
        "--summary-filename", default="truth_known_multistart_summary.json"
    )
    args = parser.parse_args()
    payload = aggregate(
        args.matrix_plan,
        args.analysis_root,
        solver=args.solver,
        success_threshold_m=args.success_threshold_m,
        summary_filename=args.summary_filename,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, payload["scenarios"])
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
