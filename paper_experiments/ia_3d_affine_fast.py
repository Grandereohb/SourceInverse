"""Same positive-quadrature affine step with one shared scalar Gram factor.

For A=mu*I+B, exp(A*s)=c(s)*I+d(s)*B. The Gram matrix of c,d
is common to both diffusion axes, so only one QR is needed per step.
"""

import numpy as np
from ia_3d_affine_transport import quadrature, horizontal_flow
from paper_affine_wind import traceless_coefficients
from paper_sqrt_affine_wind import covariance_factor


def shared_gram_step(gradient, velocity, duration, direction, order=16):
    a, b, t = np.asarray(gradient, float), np.asarray(velocity, float), np.asarray(duration, float)
    mu = np.trace(a) / 2
    traceless = a - mu * np.eye(2)
    nodes, weights = quadrature(order)
    times = t[..., None] * (nodes + 1) / 2
    cosine, sine, _, _ = traceless_coefficients(traceless, times)
    scale = np.exp(mu * times)
    c, d = scale * cosine, scale * sine
    positive_weights = t[..., None] * weights / 2
    common = covariance_factor(np.stack((c, d), axis=-2) * np.sqrt(positive_weights)[..., None, :])
    translation = (np.sum(positive_weights * c, axis=-1)[..., None] * b +
                   np.sum(positive_weights * d, axis=-1)[..., None] * (traceless @ b))
    def factor(axis):
        return np.column_stack((axis, traceless @ axis)) @ common
    divergence = 2 * mu
    z_map = np.exp(-divergence * t)
    z_variance = t.copy() if divergence == 0 else -np.expm1(-2 * divergence * t) / (2 * divergence)
    perpendicular = np.array([-direction[1], direction[0]])
    return horizontal_flow(a, t), translation, factor(direction), factor(perpendicular), z_map, z_variance
