"""Audit an observable response-count proxy across truth-known development designs."""

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


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - half, center + half


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value using fixed table margins."""
    row_one = a + b
    row_two = c + d
    column_one = a + c
    total = row_one + row_two
    lower = max(0, column_one - row_two)
    upper = min(row_one, column_one)

    def probability(x: int) -> float:
        return (
            math.comb(column_one, x)
            * math.comb(total - column_one, row_one - x)
            / math.comb(total, row_one)
        )

    observed = probability(a)
    return sum(
        probability(x)
        for x in range(lower, upper + 1)
        if probability(x) <= observed + 1e-15
    )


def _observed_response_count(
    concentration_path: Path, response_fraction: float
) -> tuple[int, float]:
    with concentration_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty concentration file: {concentration_path}")
    station_columns = [
        column for column in rows[0] if column not in {"time", "TARGET_POLLUTANT"}
    ]
    peaks = []
    for column in station_columns:
        values = sorted(float(row[column]) for row in rows)
        background = statistics.median(values[: min(3, len(values))])
        peaks.append(max(values) - background)
    global_peak = max(peaks)
    count = sum(peak > response_fraction * global_peak for peak in peaks)
    return count, global_peak


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(spec_path: Path, paper_root: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    definition = spec["proxy_definition"]
    response_fraction = float(
        definition["responsive_threshold_fraction_of_global_peak"]
    )
    low_threshold = int(definition["low_observability_threshold"])
    scenario_rows = []
    source_hashes = {"spec": _sha256(spec_path)}

    for dataset in spec["datasets"]:
        dataset_root = paper_root / dataset["root"]
        evaluation_path = dataset_root / dataset["evaluation"]
        source_hashes[f"evaluation::{dataset['id']}"] = _sha256(evaluation_path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        for source in evaluation[dataset["rows_key"]]:
            scenario_id = source["scenario_id"]
            method = source.get("method", dataset.get("method_default"))
            if method is None:
                raise ValueError(f"missing method for {dataset['id']}::{scenario_id}")
            concentration_path = dataset_root / scenario_id / "concentration.csv"
            response_count, global_peak = _observed_response_count(
                concentration_path, response_fraction
            )
            scenario_rows.append(
                {
                    "dataset_id": dataset["id"],
                    "scenario_id": scenario_id,
                    "method": method,
                    "observed_responsive_station_count": response_count,
                    "observed_global_peak_anomaly": global_peak,
                    "observability_stratum": (
                        "low_observability"
                        if response_count <= low_threshold
                        else "higher_observability"
                    ),
                    "selected_success": bool(source["selected_success"]),
                    "selected_localization_error_m": float(
                        source["selected_localization_error_m"]
                    ),
                    "concentration_sha256": _sha256(concentration_path),
                }
            )

    aggregate_rows = []
    groups = sorted({(row["dataset_id"], row["method"]) for row in scenario_rows})
    for dataset_id, method in groups:
        subset = [
            row
            for row in scenario_rows
            if row["dataset_id"] == dataset_id and row["method"] == method
        ]
        low = [row for row in subset if row["observability_stratum"] == "low_observability"]
        high = [row for row in subset if row["observability_stratum"] == "higher_observability"]
        low_success = sum(row["selected_success"] for row in low)
        high_success = sum(row["selected_success"] for row in high)
        low_ci = _wilson(low_success, len(low))
        high_ci = _wilson(high_success, len(high))
        fisher_p = _fisher_exact_two_sided(
            low_success,
            len(low) - low_success,
            high_success,
            len(high) - high_success,
        )
        direction_matches = (
            bool(low)
            and bool(high)
            and low_success / len(low) < high_success / len(high)
        )
        aggregate_rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "scenario_count": len(subset),
                "low_observability_count": len(low),
                "low_observability_success_count": low_success,
                "low_observability_success_rate": low_success / len(low) if low else math.nan,
                "low_observability_wilson95_low": low_ci[0],
                "low_observability_wilson95_high": low_ci[1],
                "higher_observability_count": len(high),
                "higher_observability_success_count": high_success,
                "higher_observability_success_rate": high_success / len(high) if high else math.nan,
                "higher_observability_wilson95_low": high_ci[0],
                "higher_observability_wilson95_high": high_ci[1],
                "fisher_exact_two_sided_p": fisher_p,
                "proxy_replication_status": "association_in_this_design_only"
                if direction_matches and fisher_p < 0.05
                else "does_not_replicate",
            }
        )

    return {
        "schema_version": 1,
        "analysis": spec["analysis_id"],
        "split": spec["split"],
        "proxy_definition": definition,
        "interpretation_limit": spec["interpretation_limit"],
        "source_hashes": source_hashes,
        "aggregate_rows": aggregate_rows,
        "scenario_rows": scenario_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--paper-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-aggregate-csv", required=True, type=Path)
    parser.add_argument("--output-scenarios-csv", required=True, type=Path)
    args = parser.parse_args()
    payload = compare(args.spec.resolve(), args.paper_root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_aggregate_csv, payload["aggregate_rows"])
    _write_csv(args.output_scenarios_csv, payload["scenario_rows"])
    for label, path in (
        ("json", args.output_json),
        ("aggregate_csv", args.output_aggregate_csv),
        ("scenarios_csv", args.output_scenarios_csv),
    ):
        print(f"{label}_sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
