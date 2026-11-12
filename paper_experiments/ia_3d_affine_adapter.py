"""IA-only point observation adapter with full 3D affine vertical transport."""

import numpy as np
import torch
from ia_metec_point_adapter import configure
from ia_3d_affine_transport import ThreeDimensionalAffineHistory


class RecordingHistory(ThreeDimensionalAffineHistory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertical_entries = []

    def relative_square_root(self, end, releases):
        state = self.relative_state(end, releases)
        self.vertical_entries.append(state[4:])
        return state[:4]


def configure_3d_affine(data, height_m, velocity, gradient, *, flow_order=16, **kwargs):
    if 'transport_history' in kwargs:
        raise ValueError('Specify the 3D affine wind directly, not a second history')
    history = RecordingHistory(np.arange(len(data['concentration']) + 1) * 60.,
        velocity, gradient, data['wind_xy'], order=flow_order)
    result = configure(data, height_m, transport_history=history, **kwargs)
    plan = result[3].plan
    if len(history.vertical_entries) != plan.groups:
        raise RuntimeError('Vertical and horizontal integration groups do not match')
    # The original time ages are no longer needed by the already-built wind/Q
    # integrals. The Gaussian kernel uses this slot as variance/(2 Dz).
    variance_age = torch.ones_like(plan.age)
    source_map = torch.ones_like(plan.age)
    for index, (mapping, variance) in enumerate(history.vertical_entries):
        if np.any(mapping <= 0) or np.any(variance <= 0):
            raise FloatingPointError('Nonpositive 3D vertical covariance or flow; no clipping')
        variance_age[index, :len(variance)] = plan.age.new_tensor(variance)
        source_map[index, :len(mapping)] = plan.age.new_tensor(mapping)
    plan.age = variance_age
    plan.vertical_source_map = source_map
    return result
