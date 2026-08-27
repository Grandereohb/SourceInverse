"""Reproducibility artifacts for inversion runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def snapshot_public_config(config_module):
    return {
        name: _json_value(getattr(config_module, name))
        for name in sorted(dir(config_module))
        if name.isupper() and not name.startswith("_")
    }


def _git_provenance(repo_root):
    repo_root = Path(repo_root)

    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_short": status.splitlines() if status else [],
    }


def save_reproducibility_bundle(
    *,
    output_dir,
    model,
    config_module,
    best_epoch,
    best_raw_loss,
    random_seed,
    source,
    normalization,
    final_transport,
    recurrent_context,
    runtime,
    q_time_series,
    copied_input_paths,
    repo_root,
):
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "best_model_state.pt"
    manifest_path = output_dir / "run_manifest.json"
    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "schema_version": 1,
            "model_name": model.__class__.__name__,
            "best_epoch": int(best_epoch),
            "best_raw_loss": float(best_raw_loss),
            "model_state_dict": cpu_state,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    input_files = {}
    for label, input_path in copied_input_paths.items():
        path = Path(input_path)
        input_files[str(label)] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    q_records = q_time_series.copy()
    if "time" in q_records.columns:
        q_records["time"] = q_records["time"].astype(str)
    manifest = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "best_epoch": int(best_epoch),
            "best_raw_loss": float(best_raw_loss),
        },
        "random_seed": int(random_seed),
        "source": _json_value(source),
        "normalization": _json_value(normalization),
        "final_transport": _json_value(final_transport),
        "recurrent_context": _json_value(recurrent_context),
        "runtime": _json_value(runtime),
        "q_time_series": q_records.to_dict(orient="records"),
        "config": snapshot_public_config(config_module),
        "inputs": input_files,
        "git": _git_provenance(repo_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
