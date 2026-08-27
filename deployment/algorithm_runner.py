from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_algorithm(
    *,
    paths: dict[str, Path],
    pollutant: str,
    algorithm_output_root: Path,
    logs_dir: Path,
    random_seed: int = 0,
    make_plots: bool = False,
    epochs: int | None = None,
) -> dict:
    logs_dir.mkdir(parents=True, exist_ok=True)
    algorithm_output_root.mkdir(parents=True, exist_ok=True)
    result_json = logs_dir / "algorithm_result.json"
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"

    command = [
        sys.executable,
        "-u",
        "-m",
        "deployment.worker_entry",
        "--sites",
        str(paths["sites"]),
        "--concentration",
        str(paths["concentration"]),
        "--wind",
        str(paths["wind"]),
        "--output-root",
        str(algorithm_output_root),
        "--result-root",
        str(algorithm_output_root),
        "--pollutant",
        pollutant,
        "--result-json",
        str(result_json),
        "--random-seed",
        str(random_seed),
    ]
    if make_plots:
        command.append("--make-plots")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
            "PINN_DEVICE": "cpu",
            "PINN_AUTO_CLOSE_PLOTS": "1",
        }
    )
    if epochs is not None:
        env["PINN_EPOCHS"] = str(max(int(epochs), 1))

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Source inversion failed with exit code {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    if not result_json.exists():
        raise RuntimeError("Source inversion completed without algorithm_result.json")
    return json.loads(result_json.read_text(encoding="utf-8"))
