from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    runpy.run_path(
        str(repo_root / "scripts" / "run_recent_leak_source_inversions.py"),
        run_name="__main__",
    )
