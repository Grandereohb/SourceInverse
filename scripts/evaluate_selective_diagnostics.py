"""Evaluate truth-blind scalar diagnostics as selective-localization scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from itertools import groupby
from pathlib import Path


DEFAULT_FEATURES = {
    "selected_best_raw_loss": "ascending",
    "selected_fit_raw_rmse": "ascending",
    "second_best_loss_relative_gap": "descending",
    "near_optimal_count": "ascending",
    "near_optimal_max_source_spread_m": "ascending",
    "selected_warning_count": "ascending",
    "selected_min_boundary_margin_m": "descending",
    "selected_dominant_station_ratio": "ascending",
    "selected_substep_cap_hit_count": "ascending",
    "selected_source_quadrature_cap_hit_count": "ascending",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def load_feature_config(path: Path) -> tuple[dict[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = list(payload["features"])
    features = {str(row["name"]): str(row["direction"]) for row in declared}
    if len(features) != len(declared):
        raise ValueError("Feature config contains duplicate feature names.")
    for feature, direction in features.items():
        if direction not in {"ascending", "descending"}:
            raise ValueError(f"Unknown direction for {feature}: {direction}")
    threshold = _finite(payload["failure_threshold_m"], "failure_threshold_m")
    if threshold <= 0.0:
        raise ValueError("failure_threshold_m must be positive.")
    return features, threshold


def _curve(
    scenarios: list[dict],
    feature: str,
    direction: str,
    failure_threshold_m: float,
) -> list[dict]:
    if direction not in {"ascending", "descending"}:
        raise ValueError(f"Unknown direction for {feature}: {direction}")
    prepared = []
    for row in scenarios:
        raw_score = _finite(row[feature], feature)
        error = _finite(
            row["selected_localization_error_m"],
            "selected_localization_error_m",
        )
        prepared.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "raw_score": raw_score,
                "sort_score": raw_score if direction == "ascending" else -raw_score,
                "error": error,
            }
        )
    prepared.sort(key=lambda row: (row["sort_score"], row["scenario_id"]))
    accepted = []
    curve = []
    for _, tied_rows in groupby(prepared, key=lambda row: row["sort_score"]):
        tied = list(tied_rows)
        accepted.extend(tied)
        errors = [row["error"] for row in accepted]
        failure_count = sum(error > failure_threshold_m for error in errors)
        threshold = tied[0]["raw_score"]
        curve.append(
            {
                "feature": feature,
                "direction": direction,
                "acceptance_rule": (
                    f"{feature} <= threshold"
                    if direction == "ascending"
                    else f"{feature} >= threshold"
                ),
                "score_threshold": threshold,
                "accepted_count": len(accepted),
                "scenario_count": len(prepared),
                "coverage": len(accepted) / len(prepared),
                "failure_count": failure_count,
                "failure_risk": failure_count / len(accepted),
                "mean_localization_error_m": statistics.fmean(errors),
                "median_localization_error_m": statistics.median(errors),
                "maximum_localization_error_m": max(errors),
                "accepted_scenario_ids": [row["scenario_id"] for row in accepted],
            }
        )
    return curve


def evaluate(
    aggregate_path: Path,
    *,
    features: dict[str, str] | None = None,
    failure_threshold_m: float | None = None,
) -> dict:
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    scenarios = list(payload["scenarios"])
    if not scenarios:
        raise ValueError("Aggregate contains no scenarios.")
    threshold = (
        _finite(payload["success_threshold_m"], "success_threshold_m")
        if failure_threshold_m is None
        else _finite(failure_threshold_m, "failure_threshold_m")
    )
    if threshold <= 0.0:
        raise ValueError("failure_threshold_m must be positive.")
    selected_features = dict(DEFAULT_FEATURES if features is None else features)
    curves = {
        feature: _curve(scenarios, feature, direction, threshold)
        for feature, direction in selected_features.items()
    }
    return {
        "schema_version": 1,
        "analysis": "development_selective_diagnostic_falsification",
        "interpretation": (
            "Scores are evaluated one at a time. Development curves may reject "
            "naive confidence proxies but must not set final test thresholds."
        ),
        "truth_usage": (
            "Localization truth is used only to evaluate risk after ranking by each "
            "declared truth-blind score."
        ),
        "aggregate_input": {
            "path": str(aggregate_path.resolve()),
            "sha256": _sha256(aggregate_path),
        },
        "scenario_count": len(scenarios),
        "failure_threshold_m": threshold,
        "feature_directions": selected_features,
        "curves": curves,
    }


def _write_csv(path: Path, curves: dict[str, list[dict]]) -> None:
    fields = [
        "feature",
        "direction",
        "acceptance_rule",
        "score_threshold",
        "accepted_count",
        "scenario_count",
        "coverage",
        "failure_count",
        "failure_risk",
        "mean_localization_error_m",
        "median_localization_error_m",
        "maximum_localization_error_m",
        "accepted_scenario_ids",
    ]
    rows = []
    for curve in curves.values():
        for row in curve:
            flattened = dict(row)
            flattened["accepted_scenario_ids"] = ";".join(
                row["accepted_scenario_ids"]
            )
            rows.append(flattened)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--failure-threshold-m", type=float)
    parser.add_argument("--feature-config", type=Path)
    args = parser.parse_args()
    if args.feature_config is not None and args.failure_threshold_m is not None:
        parser.error(
            "--failure-threshold-m cannot override a frozen --feature-config"
        )
    features = None
    threshold = args.failure_threshold_m
    if args.feature_config is not None:
        features, threshold = load_feature_config(args.feature_config)
    payload = evaluate(
        args.aggregate_json,
        features=features,
        failure_threshold_m=threshold,
    )
    if args.feature_config is not None:
        payload["feature_config"] = {
            "path": str(args.feature_config.resolve()),
            "sha256": _sha256(args.feature_config),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, payload["curves"])
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")


if __name__ == "__main__":
    main()
