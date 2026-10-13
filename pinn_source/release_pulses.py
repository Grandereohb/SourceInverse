"""Release time grids and unit test signals; no pulse inversion/search."""

import numpy as np
import torch


def build_release_aware_schedule(t_values, breakpoints, substeps=1):
    """Keep all original/substep nodes, insert release knots, return wind indices.

    Times and knots share one unit. Each returned left endpoint uses the wind
    of its original interval (right-continuous at original wind boundaries).
    This is a fixed schedule, not differentiable timing optimization.
    """
    tensor_input = isinstance(t_values, torch.Tensor)
    original = t_values.detach().cpu().numpy() if tensor_input else np.asarray(t_values)
    original = original.reshape(-1)
    times = original.astype(float)
    knots = breakpoints.detach().cpu().numpy() if isinstance(breakpoints, torch.Tensor) else np.asarray(breakpoints)
    knots = knots.astype(float).reshape(-1)
    counts = np.asarray(substeps)
    if (times.size < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(knots).all() or not np.issubdtype(counts.dtype, np.integer)):
        raise ValueError("Strictly increasing finite times, finite knots and integer substeps required.")
    counts = np.full(times.size-1, counts.item(), dtype=int) if counts.ndim == 0 else counts.reshape(-1)
    if counts.size != times.size-1 or np.any(counts < 1):
        raise ValueError("One positive substep count per original interval required.")
    precision = np.finfo(original.dtype if np.issubdtype(original.dtype, np.floating) else np.float64).eps
    tolerance = 2.*precision*max(abs(times).max(), np.ptp(times), np.finfo(float).tiny)
    if np.any(knots < times[0]-tolerance) or np.any(knots > times[-1]+tolerance):
        raise ValueError("Release breakpoints must lie inside the transport horizon.")
    nodes = np.concatenate([np.linspace(left, right, count+1)[:-1]
                            for left, right, count in zip(times[:-1], times[1:], counts)]
                           +[times[-1:]])
    extra = []
    for knot in np.unique(np.clip(knots, times[0], times[-1])):
        if np.min(abs(nodes-knot)) > tolerance and (not extra or knot-extra[-1] > tolerance):
            extra.append(knot)
    refined = np.sort(np.r_[nodes, extra])
    indices = np.clip(np.searchsorted(times, refined, side="right")-1, 0, len(times)-1)
    if tensor_input:
        dtype = t_values.dtype if torch.is_floating_point(t_values) else torch.float64
        refined = torch.as_tensor(refined, dtype=dtype, device=t_values.device)
        indices = torch.as_tensor(indices, dtype=torch.long, device=t_values.device)
    return refined, indices


def build_pulse_basis(t_values, starts, ends):
    """Unit-height triangles on explicit supports; no hidden smoothing or area normalization."""
    tensor_input = isinstance(t_values, torch.Tensor)
    times = t_values.reshape(-1) if tensor_input else torch.as_tensor(np.asarray(t_values), dtype=torch.float64).reshape(-1)
    if not torch.is_floating_point(times):
        times = times.to(torch.float32)
    left = torch.as_tensor(starts, device=times.device, dtype=times.dtype).reshape(-1)
    right = torch.as_tensor(ends, device=times.device, dtype=times.dtype).reshape(-1)
    if (times.numel() == 0 or left.numel() == 0 or left.shape != right.shape
            or not bool(torch.isfinite(times).all()) or not bool(torch.isfinite(left).all())
            or not bool(torch.isfinite(right).all()) or not bool((right > left).all())):
        raise ValueError("Finite times and nonempty, equally sized, positive-duration supports required.")
    center = .5*(left+right)
    width = .5*(right-left)
    basis = torch.clamp(1.-torch.abs(times[:, None]-center[None, :])/width[None, :], min=0.)
    return basis if tensor_input else basis.cpu().numpy()
