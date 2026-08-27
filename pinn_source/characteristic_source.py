"""Characteristic source quadrature for uniform interval transport.

This research implementation separates transport of the pre-existing field
from time integration of source releases. Each quadrature release is transported
once for its physical age, avoiding the repeated interpolation diffusion caused
by advancing the whole field for every source-integration substep.
"""

from __future__ import annotations

import math

import torch

from field import _advect_field, _diffuse_field


def source_aware_quadrature_count(
    u,
    v,
    dt,
    source_sigma,
    max_source_sigmas_per_panel=1.0,
    minimum_panels=1,
    maximum_panels=128,
):
    """Choose quadrature panels from travel distance relative to source width."""
    u_value = float(torch.as_tensor(u).detach().cpu())
    v_value = float(torch.as_tensor(v).detach().cpu())
    dt_value = max(float(torch.as_tensor(dt).detach().cpu()), 0.0)
    sigma = max(float(source_sigma), 1e-8)
    allowed = sigma * max(float(max_source_sigmas_per_panel), 1e-6)
    panel_ratio = math.hypot(u_value, v_value) * dt_value / allowed
    nearest_integer = round(panel_ratio)
    if math.isclose(panel_ratio, nearest_integer, rel_tol=1e-6, abs_tol=1e-9):
        panel_ratio = float(nearest_integer)
    requested = max(
        int(minimum_panels),
        int(math.ceil(panel_ratio)),
    )
    return min(requested, max(int(minimum_panels), int(maximum_panels))), requested


def _transport_for_age(
    field,
    x_grid,
    y_grid,
    x_mesh_flat,
    y_mesh_flat,
    u,
    v,
    diffusion,
    decay,
    age,
):
    if float(age.detach().cpu()) <= 0.0:
        return field
    transported = _advect_field(
        field,
        x_grid,
        y_grid,
        x_mesh_flat,
        y_mesh_flat,
        u,
        v,
        age,
    )
    transported = _diffuse_field(
        transported, x_grid, y_grid, diffusion, age
    )
    if decay > 0.0:
        transported = transported * torch.exp(-age * decay)
    return transported


def advance_interval_characteristic_source(
    field,
    source,
    q_samples,
    x_grid,
    y_grid,
    x_mesh_flat,
    y_mesh_flat,
    u,
    v,
    diffusion,
    decay,
    source_scale,
    dt,
):
    """Advance one interval and integrate source releases by trapezoidal quadrature.

    ``q_samples`` contains source strengths at uniformly spaced release times,
    including both interval endpoints. Its length therefore equals the number
    of quadrature panels plus one.
    """
    q_flat = torch.as_tensor(q_samples, dtype=field.dtype, device=field.device).view(-1)
    if q_flat.numel() < 2:
        raise ValueError("q_samples must include both interval endpoints")
    panels = q_flat.numel() - 1
    dt_tensor = torch.as_tensor(dt, dtype=field.dtype, device=field.device)
    existing = _transport_for_age(
        field,
        x_grid,
        y_grid,
        x_mesh_flat,
        y_mesh_flat,
        u,
        v,
        diffusion,
        decay,
        dt_tensor,
    )

    source_integral = torch.zeros_like(field)
    for index, q_value in enumerate(q_flat):
        release_fraction = index / panels
        age = dt_tensor * (1.0 - release_fraction)
        contribution = _transport_for_age(
            source,
            x_grid,
            y_grid,
            x_mesh_flat,
            y_mesh_flat,
            u,
            v,
            diffusion,
            decay,
            age,
        )
        weight = 0.5 if index in (0, panels) else 1.0
        source_integral = source_integral + weight * q_value * contribution

    source_integral = source_integral * (dt_tensor / panels) * float(source_scale)
    return existing + source_integral
