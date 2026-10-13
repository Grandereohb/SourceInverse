"""Diagnostic provenance only; never interpret unknown flags or filter observations."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


QUALITY_FILENAME = "concentration_quality.json"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_quality_record(raw_text, numeric_value, **provenance):
    raw = None if raw_text is None else str(raw_text)
    text = "" if raw is None else raw.strip()
    match = re.search(r"[\(\uff08]([A-Za-z][A-Za-z0-9_-]*)[\)\uff09]\s*$", text)
    status = match.group(1) if match else None
    relation = re.match(r"^(<=|>=|<|>|\u2264|\u2265)", text)
    finite = numeric_value is not None and math.isfinite(float(numeric_value))
    if finite:
        state = "parsed_numeric"
    elif numeric_value is not None:
        state = "nonfinite_numeric"
    elif not text or text in ("--", "---"):
        state = "missing"
    else:
        state = "unparsed"
    warnings = []
    if re.search(r"\d[eE][-+]?\d", text):
        warnings.append("Legacy numeric parser does not preserve scientific notation")
    if relation:
        warnings.append("Reported bound retained as metadata; legacy fit still uses its number")
    return {**provenance, "raw_text": raw,
        "numeric_value": float(numeric_value) if finite else None,
        "value_state": state, "status_code": status, "status_meaning": None,
        "status_state": "unknown_code" if status else "unflagged_unknown",
        "reported_relation": relation.group(1) if relation else None,
        "warnings": warnings}


def build_quality_report(records, extraction):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["station"]].append(record)
    summary = []
    for station, rows in grouped.items():
        times = defaultdict(list)
        for row in rows:
            times[row["time"]].append(row)
        duplicate = [values for values in times.values() if len(values) > 1]
        summary.append({"station": station, "source_record_count": len(rows),
            "unique_time_count": len(times),
            "value_state_counts": dict(Counter(r["value_state"] for r in rows)),
            "status_code_counts": dict(Counter(r["status_code"] or "<unflagged>" for r in rows)),
            "zero_numeric_count": sum(r["numeric_value"] == 0. for r in rows),
            "duplicate_time_count": len(duplicate),
            "mixed_status_duplicate_time_count": sum(
                len({r["status_code"] for r in values}) > 1 for values in duplicate)})
    return {"schema_version": 1, "policy": "diagnostics_only_no_filtering",
        "status_code_meanings": {}, "extraction": extraction,
        "numeric_aggregation": "Legacy per-station/time numeric mean; raw records retained separately",
        "warnings": ["Unknown codes and unflagged observations are not certified valid or invalid",
                     "Status metadata is not used by training, background estimation or scoring"],
        "station_summary": summary, "records": records}


def write_quality_report(records, concentration_path, source_path, **extraction):
    concentration_path, source_path = Path(concentration_path), Path(source_path)
    report = build_quality_report(records, {**extraction,
        "source_path": str(source_path.resolve()), "source_sha256": file_sha256(source_path),
        "concentration_path": str(concentration_path.resolve()),
        "concentration_sha256": file_sha256(concentration_path)})
    target = concentration_path.parent / QUALITY_FILENAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    return target


def archive_quality_report(concentration_path, output_dir):
    """Archive only a sidecar bound to this workbook, without blocking legacy runs."""
    concentration_path = Path(concentration_path)
    source = concentration_path.parent / QUALITY_FILENAME
    if not source.exists():
        return "absent"
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        if (not isinstance(report, dict) or report.get("schema_version") != 1
                or not isinstance(report.get("extraction"), dict)):
            return "invalid_schema"
        if report["extraction"].get("concentration_sha256") != file_sha256(concentration_path):
            return "stale_hash"
    except (OSError, UnicodeError, ValueError, TypeError):
        return "unreadable"
    destination = Path(output_dir) / QUALITY_FILENAME
    try:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    except OSError:
        return "archive_failed"
    return "archived"
