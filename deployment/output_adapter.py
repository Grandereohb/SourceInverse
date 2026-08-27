from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from .validation import TIME_FORMAT


FIELD_TIME_PATTERN = re.compile(r"(\d{8})_h(\d{2})$")


def _algorithm_version(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _read_source_strength(path: Path) -> list[dict]:
    table = pd.read_csv(path, sep="\t", header=None, names=["time", "value"])
    return [
        {
            "time": pd.Timestamp(row.time).strftime(TIME_FORMAT),
            "value": float(row.value),
            "unit": "model_unit",
        }
        for row in table.itertuples(index=False)
    ]


def _field_time(path: Path) -> str:
    match = FIELD_TIME_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Could not parse field time from filename: {path.name}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H").strftime(TIME_FORMAT)


def _read_field(path: Path) -> dict:
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["longitude", "latitude", "concentration"],
    )
    points = [
        {
            "longitude": float(row.longitude),
            "latitude": float(row.latitude),
            "concentration": float(row.concentration),
        }
        for row in table.itertuples(index=False)
    ]
    return {"time": _field_time(path), "points": points}


def build_output_payload(
    *, payload: dict,
    job_id: str,
    algorithm_result: dict,
    repo_root: Path,
) -> dict:
    case_dir = Path(algorithm_result["output_dir"])
    report_path = case_dir / "result_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exports = report["inversion_text_outputs"]

    strength_paths = [Path(path) for path in exports.get("source_strength_paths", [])]
    strength_path = next((path for path in strength_paths if path.exists()), None)
    if strength_path is None:
        raise FileNotFoundError("No source strength TXT output was found")

    selected_field_paths = []
    for alternatives in exports.get("field_paths", []):
        path = next((Path(item) for item in alternatives if Path(item).exists()), None)
        if path is None:
            raise FileNotFoundError(f"No concentration field exists for {alternatives}")
        selected_field_paths.append(path)

    return {
        "request_id": payload["request_id"],
        "job_id": job_id,
        "status": "completed",
        "pollutant": payload["pollutant"],
        "concentration_unit": payload["concentration_unit"],
        "coordinate_system": payload["coordinate_system"],
        "source": {
            "longitude": float(algorithm_result["pred_lon"]),
            "latitude": float(algorithm_result["pred_lat"]),
        },
        "release_strength": _read_source_strength(strength_path),
        "fields": [_read_field(path) for path in selected_field_paths],
        "quality": {
            "is_reasonable": bool(report.get("is_reasonable", False)),
            "warnings": [str(item) for item in report.get("warnings", [])],
        },
        "completed_at": datetime.now().strftime(TIME_FORMAT),
        "algorithm_version": _algorithm_version(repo_root),
    }
