from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    *,
    job_dir: Path,
    request_id: str,
    job_id: str,
    started_at: str,
    completed_at: str,
) -> Path:
    delivery_dir = job_dir / "delivery"
    output_path = delivery_dir / "output.json"
    manifest = {
        "request_id": request_id,
        "job_id": job_id,
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "files": {
            "output.json": {
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        },
    }
    manifest_path = delivery_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def build_zip(job_dir: Path) -> Path:
    delivery_dir = job_dir / "delivery"
    zip_path = delivery_dir / "result.zip"
    roots = [job_dir / "input", job_dir / "algorithm_output", job_dir / "logs"]
    files = [job_dir / "job.json", delivery_dir / "output.json", delivery_dir / "manifest.json"]
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(job_dir))
    return zip_path
