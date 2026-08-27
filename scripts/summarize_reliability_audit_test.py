"""Apply the preregistered reliability-audit endpoint to frozen truth summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, median


FACTOR_KEYS = (
    "input_id",
    "position_id",
    "physics_condition_id",
    "release_condition_id",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return math.nan
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (centre - radius) / denominator)


def _method_metrics(rows: list[dict]) -> dict:
    reachable = [row for row in rows if bool(row["oracle_success"])]
    selection_failures = [
        row
        for row in reachable
        if row["failure_category"] == "selection_failure_reachable"
    ]
    selected_errors = [float(row["selected_localization_error_m"]) for row in rows]
    oracle_errors = [float(row["oracle_localization_error_m"]) for row in rows]
    regrets = [float(row["selection_regret_m"]) for row in rows]
    return {
        "attempted_count": len(rows),
        "selected_success_count": sum(bool(row["selected_success"]) for row in rows),
        "oracle_reachable_count": len(reachable),
        "selection_failure_count": len(selection_failures),
        "reachability_failure_count": len(rows) - len(reachable),
        "selection_failure_rate_among_reachable": (
            len(selection_failures) / len(reachable) if reachable else math.nan
        ),
        "median_selected_error_m": median(selected_errors),
        "median_oracle_error_m": median(oracle_errors),
        "median_selection_regret_m": median(regrets),
    }


def summarize(
    protocol: dict,
    method_summaries: list[tuple[str, dict]],
    confidence: float = 0.95,
    minimum_lower_bound: float = 0.10,
) -> dict:
    primary_methods = protocol["preregistered_analysis"]["primary_methods"]
    rows_by_method: dict[str, list[dict]] = {}
    source_hashes = {}
    for source_name, payload in method_summaries:
        source_hashes[source_name] = payload.get("_source_sha256")
        for row in payload["rows"]:
            method = str(row["method"])
            if method in rows_by_method:
                raise ValueError(f"method appears in multiple summaries: {method}")
        for method in {str(row["method"]) for row in payload["rows"]}:
            rows_by_method[method] = [
                row for row in payload["rows"] if str(row["method"]) == method
            ]

    missing = [method for method in primary_methods if method not in rows_by_method]
    if missing:
        raise ValueError(f"missing preregistered primary methods: {missing}")

    method_results = {}
    stratum_rows = []
    for method, rows in sorted(rows_by_method.items()):
        metrics = _method_metrics(rows)
        metrics["one_sided_wilson_confidence"] = confidence
        metrics["selection_failure_wilson_lower_bound"] = wilson_lower_bound(
            metrics["selection_failure_count"],
            metrics["oracle_reachable_count"],
            confidence,
        )
        metrics["passes_primary_lower_bound"] = (
            method in primary_methods
            and metrics["selection_failure_wilson_lower_bound"]
            > minimum_lower_bound
        )
        method_results[method] = metrics

        for factor in FACTOR_KEYS:
            values = sorted({row.get(factor) for row in rows if row.get(factor) is not None})
            for value in values:
                subset = [row for row in rows if row.get(factor) == value]
                stratum_rows.append(
                    {
                        "method": method,
                        "factor": factor,
                        "level": value,
                        **_method_metrics(subset),
                    }
                )

    return {
        "schema_version": 1,
        "analysis": "preregistered_sumitomo_ood_reliability_audit",
        "design_id": protocol["design_id"],
        "split": protocol["split"],
        "success_threshold_m": protocol["success_threshold_m"],
        "primary_estimand": protocol["preregistered_analysis"]["primary_estimand"],
        "primary_methods": primary_methods,
        "one_sided_wilson_confidence": confidence,
        "minimum_lower_bound": minimum_lower_bound,
        "joint_confirmation_passed": all(
            method_results[method]["passes_primary_lower_bound"]
            for method in primary_methods
        ),
        "method_results": method_results,
        "strata": stratum_rows,
        "source_summary_sha256": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--surface-summary", required=True, action="append", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-lower-bound", type=float, default=0.10)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    declared_rule = protocol["preregistered_analysis"]["confirmation_rule"]
    if "one-sided 95% Wilson" not in declared_rule or "0.10" not in declared_rule:
        raise ValueError("protocol confirmation rule does not match script defaults")
    if args.confidence != 0.95 or args.minimum_lower_bound != 0.10:
        raise ValueError("changing the preregistered confidence or bound is not allowed")

    summaries = []
    for path in args.surface_summary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_source_sha256"] = _sha256(path)
        summaries.append((str(path.resolve()), payload))
    result = summarize(
        protocol,
        summaries,
        confidence=args.confidence,
        minimum_lower_bound=args.minimum_lower_bound,
    )
    result["protocol_sha256"] = _sha256(args.protocol)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["strata"][0]))
        writer.writeheader()
        writer.writerows(result["strata"])
    print(f"json={args.output_json.resolve()}")
    print(f"json_sha256={_sha256(args.output_json)}")
    print(f"csv={args.output_csv.resolve()}")
    print(f"csv_sha256={_sha256(args.output_csv)}")
    print(f"joint_confirmation_passed={result['joint_confirmation_passed']}")


if __name__ == "__main__":
    main()
