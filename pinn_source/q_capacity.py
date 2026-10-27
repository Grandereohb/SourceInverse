"""Nested nonnegative release models with a working Gaussian BIC criterion."""

import numpy as np
import torch

from variable_projection import build_q_basis, ReleaseLayout, ReleaseLayoutFamily


def nested_hat_transform(base_count, count):
    if base_count < 2 or count < 1 or count > base_count:
        raise ValueError('Positive nested capacity within a multi-hat base required')
    if count == 1:
        return np.ones((base_count, 1))
    if (base_count-1) % (count-1):
        raise ValueError('Coarse knots must belong to the fixed common base')
    clock = np.linspace(0., 1., base_count)
    knots = np.linspace(0., 1., count)
    return np.column_stack([np.interp(clock, knots, np.eye(count)[j])
                            for j in range(count)])


def gaussian_bic(sse, observations, parameters):
    if (not np.isfinite(sse) or sse < 0 or observations <= parameters or parameters < 1):
        raise ValueError('Finite SSE and more observations than parameters required')
    return float(observations*np.log(max(sse/observations, np.finfo(float).tiny))
                 + parameters*np.log(observations))


def build_capacity_family(integration_times, counts=(1, 3, 5)):
    clock = torch.as_tensor(integration_times).reshape(-1)
    if (clock.numel() < 2 or not torch.is_floating_point(clock)
            or not bool(torch.isfinite(clock).all()) or not bool((clock.diff() > 0).all())):
        raise ValueError('Ordered finite floating integration clock required')
    if (not counts or any(int(c) != c or c < 1 for c in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError('Distinct positive integer release capacities required')
    count = max(counts)
    if count > clock.numel() or any(c > 1 and (count-1) % (c-1) for c in counts):
        raise ValueError('Release capacities must be nested within the integration clock')
    common = build_q_basis(clock, count)
    nodes = torch.linspace(clock[0], clock[-1], count, dtype=clock.dtype, device=clock.device)
    layouts = []
    for c in sorted(counts):
        mapping = build_q_basis(nodes, c)
        knots = torch.linspace(clock[0], clock[-1], c, dtype=clock.dtype, device=clock.device)
        layouts.append(ReleaseLayout(f'capacity_{c}', tuple(knots.detach().cpu().tolist()),
                                     common @ mapping, mapping))
    return ReleaseLayoutFamily(common, tuple(layouts), tuple(nodes.detach().cpu().tolist()))
