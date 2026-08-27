"""Research recurrent solver using characteristic source quadrature.

The production ``field.recurrent_plume_fields`` remains unchanged. This module
provides an opt-in implementation for numerical verification and paper
experiments while the new integration scheme is being validated.
"""

from __future__ import annotations

import torch

import field as transport
from characteristic_source import (
    advance_interval_characteristic_source,
    source_aware_quadrature_count,
)


def recurrent_plume_fields_characteristic(
    model,
    sigma_src,
    *,
    initial_release_fraction=None,
    max_source_sigmas_per_panel=1.0,
    minimum_source_panels=1,
    maximum_source_panels=128,
):
    """Return fields at observation times using source-aware interval quadrature."""
    if not transport._has_recurrent_context(model):
        raise RuntimeError("Recurrent plume context has not been configured.")

    device = model.xs.device
    dtype = model.xs.dtype
    x_grid = model.recurrent_x_grid.to(device=device, dtype=dtype)
    y_grid = model.recurrent_y_grid.to(device=device, dtype=dtype)
    x_mesh = model.recurrent_x_mesh.to(device=device, dtype=dtype)
    y_mesh = model.recurrent_y_mesh.to(device=device, dtype=dtype)
    x_mesh_flat = model.recurrent_x_mesh_flat.to(device=device, dtype=dtype)
    y_mesh_flat = model.recurrent_y_mesh_flat.to(device=device, dtype=dtype)
    t_values = model.recurrent_times.to(device=device, dtype=dtype)
    u_values = model.recurrent_u.to(device=device, dtype=dtype)
    v_values = model.recurrent_v.to(device=device, dtype=dtype)

    field = torch.zeros(
        (y_grid.numel(), x_grid.numel()), dtype=dtype, device=device
    )
    source = transport._source_grid(
        model, t_values[0], x_grid, y_grid, x_mesh, y_mesh, sigma_src
    )
    diffusion = (
        float(getattr(model, "recurrent_d_min_norm", 0.0))
        + model.D() * float(getattr(model, "recurrent_d_scale_norm", 1.0))
    )
    decay = max(float(getattr(model, "recurrent_decay_norm", 0.0)), 0.0)
    source_scale = float(transport.RECURRENT_SOURCE_SCALE)

    if initial_release_fraction is None:
        initial_release_fraction = transport.RECURRENT_INITIAL_RELEASE_FRACTION
    initial_release_fraction = max(float(initial_release_fraction), 0.0)
    initial_dt = torch.as_tensor(
        getattr(model, "recurrent_initial_release_dt", 1.0),
        dtype=dtype,
        device=device,
    )
    if initial_release_fraction > 0.0:
        initial_q = model.Q(t_values[0].view(1, 1)).view(())
        field = (
            field
            + source_scale
            * initial_q
            * source
            * initial_dt
            * initial_release_fraction
        )

    fields = [field]
    selected_panels = []
    requested_panels = []
    for interval in range(t_values.numel() - 1):
        t_start = t_values[interval]
        dt = torch.clamp(t_values[interval + 1] - t_start, min=1e-6)
        selected, requested = source_aware_quadrature_count(
            u_values[interval],
            v_values[interval],
            dt,
            sigma_src,
            max_source_sigmas_per_panel=max_source_sigmas_per_panel,
            minimum_panels=minimum_source_panels,
            maximum_panels=maximum_source_panels,
        )
        selected_panels.append(selected)
        requested_panels.append(requested)
        release_fraction = torch.linspace(
            0.0, 1.0, selected + 1, dtype=dtype, device=device
        )
        release_times = t_start + dt * release_fraction
        q_samples = model.Q(release_times.view(-1, 1)).view(-1)
        field = advance_interval_characteristic_source(
            field,
            source,
            q_samples,
            x_grid,
            y_grid,
            x_mesh_flat,
            y_mesh_flat,
            u_values[interval],
            v_values[interval],
            diffusion,
            decay,
            source_scale,
            dt,
        )
        fields.append(field)

    model.research_source_quadrature_panels_per_interval = tuple(selected_panels)
    model.research_source_quadrature_required_per_interval = tuple(requested_panels)
    model.research_source_quadrature_cap_hit_count = sum(
        requested > selected
        for selected, requested in zip(selected_panels, requested_panels)
    )
    return torch.stack(fields, dim=0)
