"""Audit adaptive recurrent-substep diagnostics from saved quality reports.

The script is read-only unless --output is supplied. It intentionally separates
all recurrent reports from reports matching one declared production
configuration so that older experiments are not silently pooled.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.floor(probability * (len(ordered) - 1))
    return ordered[index]


def _load_rows(result_root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for path in result_root.rglob("result_quality_report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            recurrent = report.get("recurrent_pde") or {}
            if (report.get("model") or {}).get("field_mode") != "recurrent_pde":
                continue
            selected = list(recurrent.get("substeps_per_interval") or [])
            if not selected:
                continue
            required = list(recurrent.get("required_substeps_per_interval") or [])
            rows.append(
                {
                    "path": str(path.relative_to(result_root)),
                    "pollutant": report.get("target_pollutant"),
                    "grid_nx": int(recurrent.get("grid_nx", 0)),
                    "grid_ny": int(recurrent.get("grid_ny", 0)),
                    "maximum_substeps": int(recurrent.get("maximum_substeps", 0)),
                    "max_advection_cells_target": float(
                        recurrent.get("max_advection_cells_target", 0.0)
                    ),
                    "interval_count": len(selected),
                    "cap_hit_count": int(recurrent.get("substep_cap_hit_count", 0)),
                    "actual_max_substeps": int(
                        recurrent.get("actual_substeps_max", max(selected))
                    ),
                    "required_max_substeps": int(max(required)) if required else None,
                    "actual_max_advection_cells_per_substep": float(
                        recurrent.get("actual_max_advection_cells_per_substep", 0.0)
                    ),
                }
            )
        except Exception as exc:  # keep malformed reports visible
            errors.append(f"{path}: {exc}")
    return rows, errors


def _summarize(rows: list[dict]) -> dict:
    interval_count = sum(row["interval_count"] for row in rows)
    cap_hits = sum(row["cap_hit_count"] for row in rows)
    cap_rows = [row for row in rows if row["cap_hit_count"] > 0]
    max_cells = [row["actual_max_advection_cells_per_substep"] for row in rows]
    required = [
        row["required_max_substeps"]
        for row in rows
        if row["required_max_substeps"] is not None
    ]
    return {
        "report_count": len(rows),
        "interval_count": interval_count,
        "reports_with_cap_hit": len(cap_rows),
        "cap_hit_interval_count": cap_hits,
        "cap_hit_report_fraction": len(cap_rows) / len(rows) if rows else None,
        "cap_hit_interval_fraction": cap_hits / interval_count if interval_count else None,
        "maximum_required_substeps": max(required) if required else None,
        "maximum_actual_cells_per_substep": max(max_cells) if max_cells else None,
        "actual_max_substep_report_counts": dict(
            sorted(Counter(row["actual_max_substeps"] for row in rows).items())
        ),
        "actual_cells_per_substep_quantiles": {
            str(probability): _quantile(max_cells, probability)
            for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "cap_hit_examples": sorted(
            cap_rows,
            key=lambda row: (row["cap_hit_count"], row["required_max_substeps"] or 0),
            reverse=True,
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--grid-nx", type=int, default=36)
    parser.add_argument("--grid-ny", type=int, default=36)
    parser.add_argument("--maximum-substeps", type=int, default=3)
    parser.add_argument("--max-advection-cells", type=float, default=6.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    rows, errors = _load_rows(result_root)
    comparable = [
        row
        for row in rows
        if row["grid_nx"] == args.grid_nx
        and row["grid_ny"] == args.grid_ny
        and row["maximum_substeps"] == args.maximum_substeps
        and math.isclose(
            row["max_advection_cells_target"],
            args.max_advection_cells,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    payload = {
        "result_root": str(result_root),
        "filter": {
            "grid_nx": args.grid_nx,
            "grid_ny": args.grid_ny,
            "maximum_substeps": args.maximum_substeps,
            "max_advection_cells_target": args.max_advection_cells,
        },
        "all_recurrent_reports_with_intervals": _summarize(rows),
        "matching_configuration": _summarize(comparable),
        "parse_errors": errors,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
