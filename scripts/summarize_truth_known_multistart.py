"""Join completed inversion outputs to synthetic truth after optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _distance(left, right) -> float:
    return float(math.hypot(left[0] - right[0], left[1] - right[1]))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = 0.5 * ((cursor + 1) + end)
        for index, _ in indexed[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _load_run(manifest_path: Path, truth_xy: tuple[float, float]) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    quality_path = run_dir / "result_quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    runtime = manifest["runtime"]
    final_xy = (float(manifest["source"]["x_m"]), float(manifest["source"]["y_m"]))
    initialization = quality["source"]["initialization"]
    initial_xy = (float(initialization["x_m"]), float(initialization["y_m"]))
    return {
        "solver": str(runtime["recurrent_solver"]),
        "start_id": str(runtime["run_id"]),
        "random_seed": int(manifest["random_seed"]),
        "initial_x_m": initial_xy[0],
        "initial_y_m": initial_xy[1],
        "initial_error_m": _distance(initial_xy, truth_xy),
        "final_x_m": final_xy[0],
        "final_y_m": final_xy[1],
        "localization_error_m": _distance(final_xy, truth_xy),
        "optimizer_displacement_m": _distance(initial_xy, final_xy),
        "best_epoch": int(manifest["checkpoint"]["best_epoch"]),
        "completed_epochs": int(runtime["completed_epochs"]),
        "best_raw_loss": float(manifest["checkpoint"]["best_raw_loss"]),
        "fit_raw_rmse": float(quality["fit_raw_rmse"]),
        "training_wall_time_s": float(runtime["training_wall_time_s"]),
        "warning_count": len(quality.get("warnings", [])),
        "warnings": list(quality.get("warnings", [])),
        "run_dir": str(run_dir.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "quality_report_sha256": _sha256(quality_path),
    }


def _solver_summary(runs: list[dict]) -> dict:
    ranked_loss = sorted(runs, key=lambda row: row["best_raw_loss"])
    ranked_error = sorted(runs, key=lambda row: row["localization_error_m"])
    selected = ranked_loss[0]
    oracle = ranked_error[0]
    best_loss = selected["best_raw_loss"]
    near_limit = best_loss + max(abs(best_loss) * 0.05, 1e-12)
    near = [row for row in runs if row["best_raw_loss"] <= near_limit]
    near_points = [(row["final_x_m"], row["final_y_m"]) for row in near]
    near_spread = max(
        (_distance(left, right) for left, right in combinations(near_points, 2)),
        default=0.0,
    )
    losses = [row["best_raw_loss"] for row in runs]
    errors = [row["localization_error_m"] for row in runs]
    second_gap = (
        ranked_loss[1]["best_raw_loss"] - best_loss if len(ranked_loss) > 1 else None
    )
    return {
        "run_count": len(runs),
        "selected_start_id": selected["start_id"],
        "selected_localization_error_m": selected["localization_error_m"],
        "selected_best_raw_loss": best_loss,
        "oracle_start_id": oracle["start_id"],
        "oracle_localization_error_m": oracle["localization_error_m"],
        "selection_regret_m": selected["localization_error_m"]
        - oracle["localization_error_m"],
        "second_best_loss_gap": second_gap,
        "second_best_loss_relative_gap": (
            second_gap / max(abs(best_loss), 1e-12) if second_gap is not None else None
        ),
        "near_optimal_definition": "best_raw_loss within 5 percent of minimum",
        "near_optimal_count": len(near),
        "near_optimal_max_source_spread_m": near_spread,
        "spearman_loss_vs_localization_error": _pearson(
            _average_ranks(losses), _average_ranks(errors)
        ),
        "loss_ranking": [row["start_id"] for row in ranked_loss],
        "error_ranking": [row["start_id"] for row in ranked_error],
        "total_training_wall_time_s": sum(
            row["training_wall_time_s"] for row in runs
        ),
    }


def _evaluate_baselines(path: Path, truth_xy: tuple[float, float]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluated = []
    for estimate in payload.get("results", []):
        row = dict(estimate)
        row["localization_error_m"] = _distance(
            (float(row["source_x_m"]), float(row["source_y_m"])), truth_xy
        )
        evaluated.append(row)
    return {
        "input_path": str(path.resolve()),
        "input_sha256": _sha256(path),
        "truth_join_timing": "truth added by post-optimization evaluation",
        "results": evaluated,
    }


def summarize(
    run_root: Path,
    scenario_manifest_path: Path,
    baseline_results_paths: list[Path] | None = None,
) -> dict:
    scenario = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
    truth_xy = (float(scenario["source"]["x_m"]), float(scenario["source"]["y_m"]))
    manifests = sorted(run_root.rglob("run_manifest.json"))
    if not manifests:
        raise ValueError("No run_manifest.json files found.")
    runs = [_load_run(path, truth_xy) for path in manifests]
    keys = [(run["solver"], run["start_id"]) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate solver/start_id combinations found.")
    by_solver: dict[str, list[dict]] = {}
    for run in runs:
        by_solver.setdefault(run["solver"], []).append(run)
    result = {
        "schema_version": 1,
        "analysis": "truth_known_multistart_posthoc_evaluation",
        "truth_join_timing": "truth is read only by this post-optimization summarizer",
        "scenario": {
            "id": scenario["scenario_id"],
            "manifest_path": str(scenario_manifest_path.resolve()),
            "manifest_sha256": _sha256(scenario_manifest_path),
            "truth_model": scenario["truth_model"],
            "source_x_m": truth_xy[0],
            "source_y_m": truth_xy[1],
        },
        "run_root": str(run_root.resolve()),
        "run_count": len(runs),
        "solver_summaries": {
            solver: _solver_summary(group) for solver, group in sorted(by_solver.items())
        },
        "runs": sorted(runs, key=lambda row: (row["solver"], row["start_id"])),
    }
    if baseline_results_paths:
        evaluations = [
            _evaluate_baselines(path, truth_xy) for path in baseline_results_paths
        ]
        result["baseline_evaluations"] = {
            "sources": evaluations,
            "results": [
                row
                for evaluation in evaluations
                for row in evaluation["results"]
            ],
        }
    return result


def _write_runs_csv(path: Path, runs: list[dict]) -> None:
    fields = [
        "solver",
        "start_id",
        "initial_x_m",
        "initial_y_m",
        "initial_error_m",
        "final_x_m",
        "final_y_m",
        "localization_error_m",
        "optimizer_displacement_m",
        "best_epoch",
        "completed_epochs",
        "best_raw_loss",
        "fit_raw_rmse",
        "training_wall_time_s",
        "warning_count",
        "run_dir",
        "manifest_sha256",
        "quality_report_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--scenario-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--baseline-results", nargs="+", type=Path)
    args = parser.parse_args()
    payload = summarize(
        args.run_root, args.scenario_manifest, args.baseline_results
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_runs_csv(args.output_csv, payload["runs"])
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(json.dumps(payload["solver_summaries"], ensure_ascii=False, indent=2))
    if "baseline_evaluations" in payload:
        print(json.dumps(payload["baseline_evaluations"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
