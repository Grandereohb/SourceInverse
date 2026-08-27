"""Summarize truth-blind source drift across frozen transport perturbation matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(math.hypot(left[0] - right[0], left[1] - right[1]))


def _parse_matrix_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Matrix spec must be LABEL=PATH.")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
        raise ValueError(f"Invalid matrix label: {label!r}")
    return label, Path(raw_path)


def _comparable_plan(plan: dict) -> dict:
    physics = dict(plan["physics_settings"])
    physics.pop("wind_scale", None)
    return {
        "batch_manifest_sha256": str(plan["batch_manifest"]["sha256"]),
        "scenario_ids": [str(row["scenario_id"]) for row in plan["scenarios"]],
        "solvers": list(plan["solvers"]),
        "epochs": int(plan["epochs"]),
        "seed": int(plan["seed"]),
        "q_mode": str(plan["q_mode"]),
        "source_position_mode": str(plan["source_position_mode"]),
        "physics_except_wind_scale": physics,
    }


def _load_matrix(root: Path, label: str, solver: str, start_id: str) -> dict:
    plan_path = root / "matrix_experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = {}
    for expected in plan["scenarios"]:
        scenario_id = str(expected["scenario_id"])
        manifests = []
        for path in (root / scenario_id).rglob("run_manifest.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            runtime = payload["runtime"]
            if (
                str(runtime["recurrent_solver"]) == solver
                and str(runtime["run_id"]) == start_id
            ):
                manifests.append((path, payload))
        if len(manifests) != 1:
            raise ValueError(
                f"Expected one {solver}/{start_id} run for {scenario_id} in {root}, "
                f"found {len(manifests)}."
            )
        manifest_path, manifest = manifests[0]
        quality_path = manifest_path.parent / "result_quality_report.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        initialization = quality["source"]["initialization"]
        rows[scenario_id] = {
            "source_xy": (
                float(manifest["source"]["x_m"]),
                float(manifest["source"]["y_m"]),
            ),
            "initial_xy": (
                float(initialization["x_m"]),
                float(initialization["y_m"]),
            ),
            "best_raw_loss": float(manifest["checkpoint"]["best_raw_loss"]),
            "fit_raw_rmse": float(quality["fit_raw_rmse"]),
            "training_wall_time_s": float(
                manifest["runtime"]["training_wall_time_s"]
            ),
            "input_digests": {
                key: str(value["sha256"])
                for key, value in manifest["inputs"].items()
            },
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "quality_report_sha256": _sha256(quality_path),
        }
    return {
        "label": label,
        "root": str(root.resolve()),
        "plan": plan,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "wind_scale": float(plan["physics_settings"]["wind_scale"]),
        "rows": rows,
    }


def summarize(
    matrix_specs: list[tuple[str, Path]],
    truth_aggregate_path: Path,
    *,
    baseline_label: str,
    solver: str = "production",
    start_id: str = "heuristic",
) -> dict:
    labels = [label for label, _ in matrix_specs]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("At least two uniquely labelled matrices are required.")
    if baseline_label not in labels:
        raise ValueError("baseline_label must name one of the matrices.")
    matrices = [
        _load_matrix(root, label, solver, start_id) for label, root in matrix_specs
    ]
    comparable = _comparable_plan(matrices[0]["plan"])
    for matrix in matrices[1:]:
        if _comparable_plan(matrix["plan"]) != comparable:
            raise ValueError(
                f"Matrix {matrix['label']} differs in more than wind scale/design."
            )
    wind_scales = [matrix["wind_scale"] for matrix in matrices]
    if len(wind_scales) != len(set(wind_scales)):
        raise ValueError("Every matrix must have a distinct wind scale.")
    truth_aggregate = json.loads(truth_aggregate_path.read_text(encoding="utf-8"))
    truth_rows = {
        str(row["scenario_id"]): row for row in truth_aggregate["scenarios"]
    }
    scenario_ids = comparable["scenario_ids"]
    if set(truth_rows) != set(scenario_ids):
        raise ValueError("Truth aggregate scenario IDs do not match matrix plans.")
    by_label = {matrix["label"]: matrix for matrix in matrices}
    baseline = by_label[baseline_label]
    rows = []
    for scenario_id in scenario_ids:
        scenario_runs = {
            matrix["label"]: matrix["rows"][scenario_id] for matrix in matrices
        }
        initial_points = [run["initial_xy"] for run in scenario_runs.values()]
        if max(
            (_distance(left, right) for left, right in combinations(initial_points, 2)),
            default=0.0,
        ) > 1e-6:
            raise ValueError(f"Initial coordinates differ for {scenario_id}.")
        input_sets = {
            json.dumps(run["input_digests"], sort_keys=True)
            for run in scenario_runs.values()
        }
        if len(input_sets) != 1:
            raise ValueError(f"Input hashes differ for {scenario_id}.")
        points = {label: run["source_xy"] for label, run in scenario_runs.items()}
        max_pairwise = max(
            (_distance(left, right) for left, right in combinations(points.values(), 2)),
            default=0.0,
        )
        baseline_point = points[baseline_label]
        max_from_baseline = max(
            _distance(baseline_point, point) for point in points.values()
        )
        truth = truth_rows[scenario_id]
        row = {
            "scenario_id": scenario_id,
            "selected_localization_error_m": float(
                truth["selected_localization_error_m"]
            ),
            "selected_success": bool(truth["selected_success"]),
            "failure_category": str(truth["failure_category"]),
            "wind_scale_max_pairwise_source_drift_m": max_pairwise,
            "wind_scale_max_baseline_source_drift_m": max_from_baseline,
            "initial_x_m": initial_points[0][0],
            "initial_y_m": initial_points[0][1],
            "total_perturbation_training_wall_time_s": sum(
                run["training_wall_time_s"] for run in scenario_runs.values()
            ),
        }
        for label, run in scenario_runs.items():
            row[f"{label}_wind_scale"] = by_label[label]["wind_scale"]
            row[f"{label}_source_x_m"] = run["source_xy"][0]
            row[f"{label}_source_y_m"] = run["source_xy"][1]
            row[f"{label}_best_raw_loss"] = run["best_raw_loss"]
            row[f"{label}_fit_raw_rmse"] = run["fit_raw_rmse"]
            row[f"{label}_manifest_sha256"] = run["manifest_sha256"]
        rows.append(row)
    return {
        "schema_version": 1,
        "analysis": "truth_blind_transport_perturbation_drift_with_posthoc_risk_join",
        "truth_usage": (
            "source truth is read only from the completed multistart aggregate after "
            "all perturbation inversions; it does not affect starts or optimization"
        ),
        "diagnostic_scope": (
            "wind-scale drift is computed from a shared heuristic start and evaluated "
            "as an event-level diagnostic for the multistart-selected point"
        ),
        "solver": solver,
        "start_id": start_id,
        "baseline_label": baseline_label,
        "success_threshold_m": float(truth_aggregate["success_threshold_m"]),
        "scenario_count": len(rows),
        "truth_aggregate": {
            "path": str(truth_aggregate_path.resolve()),
            "sha256": _sha256(truth_aggregate_path),
        },
        "matrices": [
            {
                "label": matrix["label"],
                "root": matrix["root"],
                "wind_scale": matrix["wind_scale"],
                "plan_path": matrix["plan_path"],
                "plan_sha256": matrix["plan_sha256"],
            }
            for matrix in matrices
        ],
        "scenarios": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--truth-aggregate", required=True, type=Path)
    parser.add_argument("--solver", default="production")
    parser.add_argument("--start-id", default="heuristic")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    specs = [_parse_matrix_spec(value) for value in args.matrix]
    payload = summarize(
        specs,
        args.truth_aggregate,
        baseline_label=args.baseline_label,
        solver=args.solver,
        start_id=args.start_id,
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


if __name__ == "__main__":
    main()
