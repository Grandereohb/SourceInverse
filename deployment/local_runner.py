from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .algorithm_runner import REPO_ROOT, run_algorithm
from .input_adapter import load_and_validate_input, write_algorithm_inputs
from .output_adapter import build_output_payload
from .packager import build_zip, write_manifest
from .validation import TIME_FORMAT


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._") or "job"


def _write_job(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_local_job(
    *,
    input_path: str | Path,
    work_root: str | Path,
    epochs: int | None = None,
    random_seed: int = 0,
    make_plots: bool = False,
    job_id: str | None = None,
    job_dir: str | Path | None = None,
    completion_status: str = "completed",
) -> dict:
    started_at = datetime.now().strftime(TIME_FORMAT)
    payload = load_and_validate_input(input_path)
    job_id = job_id or str(uuid.uuid4())
    job_dir = (
        Path(job_dir).resolve()
        if job_dir is not None
        else Path(work_root).resolve() / f"{_safe_name(payload['request_id'])}_{job_id[:8]}"
    )
    input_dir = job_dir / "input"
    output_root = job_dir / "algorithm_output"
    logs_dir = job_dir / "logs"
    delivery_dir = job_dir / "delivery"
    for directory in (input_dir, output_root, logs_dir, delivery_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_copy = input_dir / "input.json"
    if Path(input_path).resolve() != input_copy.resolve():
        shutil.copy2(input_path, input_copy)
    job_path = job_dir / "job.json"
    job = {
        "request_id": payload["request_id"],
        "job_id": job_id,
        "status": "validating",
        "started_at": started_at,
        "completed_at": None,
        "error": None,
    }
    _write_job(job_path, job)

    try:
        paths = write_algorithm_inputs(payload, input_dir)
        job["status"] = "running"
        _write_job(job_path, job)
        algorithm_result = run_algorithm(
            paths=paths,
            pollutant=payload["pollutant"],
            algorithm_output_root=output_root,
            logs_dir=logs_dir,
            random_seed=random_seed,
            make_plots=make_plots,
            epochs=epochs,
        )
        job["status"] = "packaging"
        _write_job(job_path, job)
        output = build_output_payload(
            payload=payload,
            job_id=job_id,
            algorithm_result=algorithm_result,
            repo_root=REPO_ROOT,
        )
        output_path = delivery_dir / "output.json"
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        completed_at = output["completed_at"]
        job.update({"status": completion_status, "completed_at": completed_at})
        _write_job(job_path, job)
        manifest_path = write_manifest(
            job_dir=job_dir,
            request_id=payload["request_id"],
            job_id=job_id,
            started_at=started_at,
            completed_at=completed_at,
        )
        zip_path = build_zip(job_dir)
        return {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "output_json": str(output_path),
            "manifest_json": str(manifest_path),
            "zip_path": str(zip_path),
        }
    except Exception as exc:
        job.update(
            {
                "status": "failed",
                "completed_at": datetime.now().strftime(TIME_FORMAT),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_job(job_path, job)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local deployment job")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--work-root", default="deployment_runs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_local_job(
        input_path=args.input,
        work_root=args.work_root,
        epochs=args.epochs,
        random_seed=args.random_seed,
        make_plots=args.make_plots,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
