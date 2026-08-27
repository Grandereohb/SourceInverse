"""Summarize a crossed Gaussian source-inversion mechanism experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return 0.5 * (
        ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]
    )


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - half, center + half


def _exact_paired_binary_p(first: int, second: int) -> float:
    discordant = int(first) + int(second)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(first, second) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def _bootstrap_median_ci(
    values: list[float], label: str, iterations: int = 20000
) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(_median(sample))
    estimates.sort()
    return (
        estimates[int(0.025 * iterations)],
        estimates[int(0.975 * iterations) - 1],
    )


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_key, subset in sorted(grouped.items()):
        selected = sum(bool(row["selected_success"]) for row in subset)
        oracle = sum(bool(row["oracle_success"]) for row in subset)
        lower, upper = _wilson(selected, len(subset))
        output.append(
            {
                **dict(zip(keys, group_key)),
                "scenario_count": len(subset),
                "selected_success_count": selected,
                "selected_success_rate": selected / len(subset),
                "selected_success_wilson95_low": lower,
                "selected_success_wilson95_high": upper,
                "oracle_success_count": oracle,
                "selection_failure_count": sum(
                    row["failure_category"] == "selection_failure_reachable"
                    for row in subset
                ),
                "reachability_failure_count": sum(
                    row["failure_category"] == "candidate_reachability_failure"
                    for row in subset
                ),
                "selected_error_median_m": _median(
                    [row["selected_localization_error_m"] for row in subset]
                ),
                "oracle_error_median_m": _median(
                    [row["oracle_localization_error_m"] for row in subset]
                ),
                "selection_regret_median_m": _median(
                    [row["selection_regret_m"] for row in subset]
                ),
                "responsive_station_count_median": _median(
                    [row["responsive_station_count"] for row in subset]
                ),
                "dominant_signal_energy_ratio_median": _median(
                    [row["dominant_signal_energy_ratio"] for row in subset]
                ),
            }
        )
    return output


def summarize(
    gaussian_path: Path, quality_csv: Path, reference_condition: str = "matched_default"
) -> dict:
    gaussian = json.loads(gaussian_path.read_text(encoding="utf-8"))
    with quality_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        quality = {row["scenario_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for source in gaussian["rows"]:
        audit = quality[source["scenario_id"]]
        row = dict(source)
        row["responsive_station_count"] = int(
            audit["responsive_station_count_peak_gt_5pct_global"]
        )
        row["dominant_signal_energy_ratio"] = float(
            audit["dominant_signal_energy_ratio"]
        )
        row["response_stratum"] = (
            "one_responsive_station"
            if row["responsive_station_count"] <= 1
            else "multiple_responsive_stations"
        )
        rows.append(row)

    factor_rows = _group_rows(rows, ("method", "physics_condition_id"))
    response_rows = _group_rows(rows, ("method", "response_stratum"))

    paired_rows = []
    conditions = sorted({row["physics_condition_id"] for row in rows})
    reference = reference_condition
    if reference not in conditions:
        raise ValueError(
            f"reference condition {reference!r} not found; available={conditions}"
        )
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        by_cell = {
            (row["input_id"], row["position_id"], row["physics_condition_id"]): row
            for row in method_rows
        }
        for condition in conditions:
            if condition == reference:
                continue
            differences = []
            success_changes = []
            selection_failure_changes = []
            for input_id in sorted({row["input_id"] for row in method_rows}):
                for position_id in sorted({row["position_id"] for row in method_rows}):
                    before = by_cell[(input_id, position_id, reference)]
                    after = by_cell[(input_id, position_id, condition)]
                    differences.append(
                        float(after["selected_localization_error_m"])
                        - float(before["selected_localization_error_m"])
                    )
                    success_changes.append(
                        int(bool(after["selected_success"]))
                        - int(bool(before["selected_success"]))
                    )
                    selection_failure_changes.append(
                        int(
                            after["failure_category"]
                            == "selection_failure_reachable"
                        )
                        - int(
                            before["failure_category"]
                            == "selection_failure_reachable"
                        )
                    )
            success_lost = sum(value < 0 for value in success_changes)
            success_gained = sum(value > 0 for value in success_changes)
            error_worse = sum(value > 0.0 for value in differences)
            error_better = sum(value < 0.0 for value in differences)
            ci_low, ci_high = _bootstrap_median_ci(
                differences, f"{method}::{reference}::{condition}"
            )
            paired_rows.append(
                {
                    "method": method,
                    "reference_condition": reference,
                    "contrast_condition": condition,
                    "paired_cell_count": len(differences),
                    "median_error_difference_m": _median(differences),
                    "median_error_difference_bootstrap95_low_m": ci_low,
                    "median_error_difference_bootstrap95_high_m": ci_high,
                    "mean_error_difference_m": sum(differences) / len(differences),
                    "error_worse_count": error_worse,
                    "error_better_count": error_better,
                    "error_sign_test_exact_two_sided_p": _exact_paired_binary_p(
                        error_worse, error_better
                    ),
                    "selected_success_lost_count": success_lost,
                    "selected_success_gained_count": success_gained,
                    "mcnemar_exact_two_sided_p": _exact_paired_binary_p(
                        success_lost, success_gained
                    ),
                    "net_selected_success_change": sum(success_changes),
                    "net_selection_failure_change": sum(selection_failure_changes),
                }
            )
    return {
        "schema_version": 1,
        "analysis": "crossed_gaussian_failure_mechanism_summary",
        "source_hashes": {
            "gaussian_truth_evaluation": _sha256(gaussian_path),
            "quality_audit_csv": _sha256(quality_csv),
        },
        "success_threshold_m": gaussian["success_threshold_m"],
        "interpretation_limits": (
            "Development-only crossed effects. Physics contrasts are paired by "
            "layout and normalized source position. Response strata are descriptive, "
            "not causal or calibrated reliability scores."
        ),
        "factor_rows": factor_rows,
        "response_stratum_rows": response_rows,
        "paired_contrast_rows": paired_rows,
        "scenario_rows": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-json", required=True, type=Path)
    parser.add_argument("--quality-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-factor-csv", required=True, type=Path)
    parser.add_argument("--output-response-csv", required=True, type=Path)
    parser.add_argument("--output-contrasts-csv", required=True, type=Path)
    parser.add_argument("--reference-condition", default="matched_default")
    args = parser.parse_args()
    payload = summarize(
        args.gaussian_json,
        args.quality_csv,
        reference_condition=args.reference_condition,
    )
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_factor_csv, payload["factor_rows"])
    _write_csv(args.output_response_csv, payload["response_stratum_rows"])
    _write_csv(args.output_contrasts_csv, payload["paired_contrast_rows"])
    for label, path in (
        ("json", args.output_json),
        ("factor_csv", args.output_factor_csv),
        ("response_csv", args.output_response_csv),
        ("contrasts_csv", args.output_contrasts_csv),
    ):
        print(f"{label}_sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
