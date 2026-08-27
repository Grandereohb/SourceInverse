"""Development benchmark for production and characteristic recurrent solvers."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import field as production  # noqa: E402
from research_recurrent import recurrent_plume_fields_characteristic  # noqa: E402


class BenchmarkModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.xs = torch.nn.Parameter(torch.tensor(-0.25, dtype=dtype))
        self.ys = torch.nn.Parameter(torch.tensor(0.05, dtype=dtype))
        self.raw_d = torch.nn.Parameter(torch.tensor(-4.0, dtype=dtype))
        self.log_q = torch.nn.Parameter(torch.tensor(0.1, dtype=dtype))

    def D(self):
        return torch.nn.functional.softplus(self.raw_d)

    def Q(self, t):
        return torch.exp(self.log_q) * torch.ones_like(t)

    def source_xy(self, t):
        return self.xs.expand_as(t), self.ys.expand_as(t)


def _time_call(callable_, repeats, backward):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fields = callable_()
        if backward:
            loss = fields[-1].square().mean()
            loss.backward()
            for parameter in callable_.model.parameters():
                parameter.grad = None
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "repeats": repeats,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=36)
    parser.add_argument("--times", type=int, default=13)
    parser.add_argument("--velocity", type=float, default=2.4)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--forward-repeats", type=int, default=20)
    parser.add_argument("--backward-repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    model = BenchmarkModel()
    times = torch.linspace(0.0, 1.0, args.times).tolist()
    production.configure_recurrent_context(
        model,
        x_min=-0.5,
        x_max=0.5,
        y_min=-0.5,
        y_max=0.5,
        t_values=times,
        u_values=[args.velocity] * args.times,
        v_values=[0.0] * args.times,
        d_min_norm=0.0,
        d_scale_norm=1.0,
        decay_norm=0.5,
        nx=args.grid,
        ny=args.grid,
    )

    def production_call():
        with patch.object(production, "RECURRENT_INITIAL_RELEASE_FRACTION", 0.0):
            return production.recurrent_plume_fields(model, args.sigma)

    def characteristic_call():
        return recurrent_plume_fields_characteristic(
            model,
            args.sigma,
            initial_release_fraction=0.0,
            max_source_sigmas_per_panel=1.0,
        )

    production_call.model = model
    characteristic_call.model = model

    for _ in range(3):
        production_call()
        characteristic_call()

    with torch.no_grad():
        production_fields = production_call()
        characteristic_fields = characteristic_call()
        relative_field_difference = float(
            torch.linalg.vector_norm(production_fields - characteristic_fields)
            / torch.linalg.vector_norm(characteristic_fields)
        )

    production_forward = _time_call(
        production_call, args.forward_repeats, backward=False
    )
    characteristic_forward = _time_call(
        characteristic_call, args.forward_repeats, backward=False
    )
    production_backward = _time_call(
        production_call, args.backward_repeats, backward=True
    )
    characteristic_backward = _time_call(
        characteristic_call, args.backward_repeats, backward=True
    )

    payload = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "problem": {
            "grid": args.grid,
            "observation_times": args.times,
            "intervals": args.times - 1,
            "velocity": args.velocity,
            "source_sigma": args.sigma,
            "production_substeps_per_interval": list(
                model.recurrent_substeps_per_interval
            ),
            "characteristic_panels_per_interval": list(
                model.research_source_quadrature_panels_per_interval
            ),
        },
        "production": {
            "forward": production_forward,
            "forward_backward": production_backward,
        },
        "characteristic": {
            "forward": characteristic_forward,
            "forward_backward": characteristic_backward,
        },
        "ratios": {
            "characteristic_to_production_forward": characteristic_forward[
                "median_seconds"
            ]
            / production_forward["median_seconds"],
            "characteristic_to_production_forward_backward": characteristic_backward[
                "median_seconds"
            ]
            / production_backward["median_seconds"],
        },
        "production_vs_characteristic_relative_field_difference": relative_field_difference,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
