"""Compare recurrent source integration with a characteristic-time reference.

The verification problem uses a stationary Gaussian source, constant positive
x wind, zero diffusion and zero decay. Its final field is the time integral of
the source kernel translated along characteristics. The reference evaluates
that integral directly, avoiding repeated interpolation of the evolving field.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import field as transport  # noqa: E402
from characteristic_source import (  # noqa: E402
    advance_interval_characteristic_source,
    source_aware_quadrature_count,
)


def _integral(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    dx = grid[1] - grid[0]
    return torch.sum(field) * dx * dx


def _source(grid: torch.Tensor, x_source: float, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    source = torch.exp(-((xx - x_source) ** 2 + yy**2) / (2.0 * sigma**2))
    return source / _integral(source, grid)


def _simulate(
    grid_size: int,
    substeps: int,
    velocity: float,
    source_x: float,
    source_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(-0.5, 0.5, grid_size, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    source = _source(grid, source_x, source_sigma)
    field = torch.zeros_like(source)
    dt = torch.tensor(1.0 / substeps, dtype=grid.dtype)
    one = torch.tensor(1.0, dtype=grid.dtype)
    zero = torch.tensor(0.0, dtype=grid.dtype)
    for _ in range(substeps):
        field = transport._advance_recurrent_step(
            field,
            source,
            one,
            source,
            one,
            grid,
            grid,
            xx.reshape(-1),
            yy.reshape(-1),
            torch.tensor(velocity, dtype=grid.dtype),
            zero,
            zero,
            0.0,
            1.0,
            dt,
        )
    return grid, field


def _characteristic_reference(
    grid_size: int,
    velocity: float,
    source_x: float,
    source_sigma: float,
    time_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(-0.5, 0.5, grid_size, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    base = torch.exp(-((xx - source_x) ** 2 + yy**2) / (2.0 * source_sigma**2))
    source_mass = _integral(base, grid)
    field = torch.zeros_like(base)
    for index in range(time_samples):
        release_time = index / (time_samples - 1)
        age = 1.0 - release_time
        x_back = xx - velocity * age
        weight = 0.5 if index in (0, time_samples - 1) else 1.0
        within_source_domain = (x_back >= -0.5) & (x_back <= 0.5)
        translated = torch.where(
            within_source_domain,
            torch.exp(
                -((x_back - source_x) ** 2 + yy**2) / (2.0 * source_sigma**2)
            )
            / source_mass,
            0.0,
        )
        field += weight * translated
    field /= time_samples - 1
    return grid, field


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=[36, 72, 144])
    parser.add_argument("--velocity", type=float, default=0.8)
    parser.add_argument("--source-x", type=float, default=-0.4)
    parser.add_argument("--source-sigma", type=float, default=0.05)
    parser.add_argument("--target-cells", type=float, default=6.0)
    parser.add_argument("--substep-cap", type=int, default=3)
    parser.add_argument("--reference-time-samples", type=int, default=4001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for grid_size in args.grids:
        displacement_cells = abs(args.velocity) * (grid_size - 1)
        required = max(1, math.ceil(displacement_cells / args.target_cells - 1e-12))
        selected = min(required, args.substep_cap)
        grid, reference = _characteristic_reference(
            grid_size,
            args.velocity,
            args.source_x,
            args.source_sigma,
            args.reference_time_samples,
        )
        reference_norm = torch.linalg.vector_norm(reference)
        reference_mass = float(_integral(reference, grid))
        for name, substeps in (
            ("production_capped", selected),
            ("uncapped_target", required),
        ):
            _, numerical = _simulate(
                grid_size,
                substeps,
                args.velocity,
                args.source_x,
                args.source_sigma,
            )
            rows.append(
                {
                    "grid": grid_size,
                    "scheme": name,
                    "displacement_cells_per_interval": displacement_cells,
                    "substeps": substeps,
                    "cells_per_substep": displacement_cells / substeps,
                    "relative_l2": float(
                        torch.linalg.vector_norm(numerical - reference) / reference_norm
                    ),
                    "mass": float(_integral(numerical, grid)),
                    "reference_mass": reference_mass,
                }
            )

        source_panels, source_panels_requested = source_aware_quadrature_count(
            args.velocity,
            0.0,
            1.0,
            args.source_sigma,
            max_source_sigmas_per_panel=1.0,
            maximum_panels=128,
        )
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        source = _source(grid, args.source_x, args.source_sigma)
        characteristic = advance_interval_characteristic_source(
            field=torch.zeros_like(source),
            source=source,
            q_samples=torch.ones(source_panels + 1, dtype=grid.dtype),
            x_grid=grid,
            y_grid=grid,
            x_mesh_flat=xx.reshape(-1),
            y_mesh_flat=yy.reshape(-1),
            u=torch.tensor(args.velocity, dtype=grid.dtype),
            v=torch.tensor(0.0, dtype=grid.dtype),
            diffusion=torch.tensor(0.0, dtype=grid.dtype),
            decay=0.0,
            source_scale=1.0,
            dt=torch.tensor(1.0, dtype=grid.dtype),
        )
        rows.append(
            {
                "grid": grid_size,
                "scheme": "characteristic_source_aware",
                "displacement_cells_per_interval": displacement_cells,
                "substeps": None,
                "source_quadrature_panels": source_panels,
                "source_quadrature_panels_requested": source_panels_requested,
                "source_sigmas_per_panel": abs(args.velocity)
                / (source_panels * args.source_sigma),
                "relative_l2": float(
                    torch.linalg.vector_norm(characteristic - reference) / reference_norm
                ),
                "mass": float(_integral(characteristic, grid)),
                "reference_mass": reference_mass,
            }
        )

    payload = {
        "problem": {
            "domain": [-0.5, 0.5, -0.5, 0.5],
            "duration": 1.0,
            "velocity": [args.velocity, 0.0],
            "source": [args.source_x, 0.0],
            "source_sigma": args.source_sigma,
            "source_strength": 1.0,
            "diffusion": 0.0,
            "decay": 0.0,
            "target_cells": args.target_cells,
            "substep_cap": args.substep_cap,
            "reference_time_samples": args.reference_time_samples,
        },
        "results": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
