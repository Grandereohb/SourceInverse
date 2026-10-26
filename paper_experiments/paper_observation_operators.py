"""Experimental beam-average operator; no point-sensor substitution.

Coordinates may be2D or3D. Weights produce a path MEAN, not an integral in
concentration times length. This does not supply a vertical transport model.
"""

import numpy as np
import torch
from scipy.special import roots_legendre


def beam_quadrature(starts, ends, order=32):
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    if (starts.ndim != 2 or starts.shape[1] not in (2,3) or starts.shape != ends.shape
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.any(np.linalg.norm(ends-starts, axis=1) <= 0)
            or int(order) != order or order < 2):
        raise ValueError('Finite nonzero matching2D/3D beams and order>=2 required')
    nodes, weights = roots_legendre(int(order))
    fraction = (nodes+1)*0.5
    points = starts[:,None,:] + (ends-starts)[:,None,:]*fraction[None,:,None]
    return points, np.broadcast_to(weights[None,:]*0.5, points.shape[:2]).copy()


def beam_mean_response(point_response, weights):
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(1),1.,rtol=0.,atol=1e-12)
            or point_response.ndim != 2 or point_response.shape[0] != weights.size):
        raise ValueError('Responses ordered by beam then node and normalized nonnegative weights required')
    if isinstance(point_response, torch.Tensor):
        w = torch.as_tensor(weights, device=point_response.device, dtype=point_response.dtype)
        return (point_response.reshape(*weights.shape,-1)*w[:,:,None]).sum(1)
    return (np.asarray(point_response).reshape(*weights.shape,-1)*weights[:,:,None]).sum(1)
