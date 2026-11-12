"""Research-only 3D incompressible completion of measured horizontal affine wind.

Horizontal u=b+A*x, vertical w=-tr(A)*z. This is a kinematic closure,
not an observation of vertical air motion. Ground is invariant under this flow.
"""

from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from paper_affine_wind import traceless_coefficients
from paper_sqrt_affine_wind import covariance_factor


@lru_cache(maxsize=4)
def quadrature(order):
    return leggauss(order)


def horizontal_flow(gradient, duration):
    a = np.asarray(gradient, float)
    t = np.asarray(duration, float)
    half_trace = np.trace(a) / 2
    traceless = a - half_trace * np.eye(2)
    cosine, sine, _, _ = traceless_coefficients(traceless, t)
    return np.exp(half_trace * t)[..., None, None] * (
        cosine[..., None, None] * np.eye(2) + sine[..., None, None] * traceless)


def general_step(gradient, velocity, duration, direction, order=16):
    """Positive quadrature Gram factors and exact vertical OU-free propagation."""
    a, b = np.asarray(gradient, float), np.asarray(velocity, float)
    t = np.asarray(duration, float)
    if (a.shape != (2, 2) or b.shape != (2,) or np.any(t < 0) or
            not all(np.isfinite(v).all() for v in [a, b, t])):
        raise ValueError('Finite horizontal affine coefficients and nonnegative durations required')
    nodes, weights = quadrature(order)
    times = t[..., None] * (nodes + 1) / 2
    transforms = horizontal_flow(a, times)
    positive_weights = t[..., None] * weights / 2
    translation = np.einsum('...nij,j,...n->...i', transforms, b, positive_weights)
    perpendicular = np.array([-direction[1], direction[0]])
    def factor(axis):
        columns = np.einsum('...nij,j->...in', transforms, axis) * np.sqrt(positive_weights)[..., None, :]
        return covariance_factor(columns)
    divergence = np.trace(a)
    z_map = np.exp(-divergence * t)
    if divergence == 0:
        z_half_variance = t.copy()
    else:
        z_half_variance = -np.expm1(-2 * divergence * t) / (2 * divergence)
    return horizontal_flow(a, t), translation, factor(direction), factor(perpendicular), z_map, z_half_variance


class ThreeDimensionalAffineHistory:
    def __init__(self, edges, velocity, gradient, diffusion_direction, order=16):
        self.edges = np.asarray(edges, float)
        self.velocity = np.asarray(velocity, float)
        self.gradient = np.asarray(gradient, float)
        wind = np.asarray(diffusion_direction, float)
        if (self.edges.ndim != 1 or self.edges[0] != 0 or np.any(np.diff(self.edges) <= 0) or
                self.velocity.shape != (len(self.edges) - 1, 2) or self.gradient.shape != (len(self.velocity), 2, 2) or
                wind.shape != self.velocity.shape or not all(np.isfinite(v).all() for v in
                    [self.edges, self.velocity, self.gradient, wind])):
            raise ValueError('Matching finite affine wind records required')
        speed = np.linalg.norm(wind, axis=1)
        self.direction = wind / np.where(speed > 0, speed, 1.)[:, None]
        self.direction[speed == 0] = [1., 0.]
        self.order = order

    def relative_state(self, end, releases):
        releases = np.asarray(releases, float)
        if (releases.ndim != 1 or not np.isfinite(releases).all() or
                np.any(releases < 0) or np.any(releases >= end) or end > self.edges[-1]):
            raise ValueError('Release times must precede end within the wind clock')
        maps = np.zeros((len(releases), 2, 2))
        shifts = np.zeros((len(releases), 2))
        parallel, perpendicular = np.zeros_like(maps), np.zeros_like(maps)
        vertical_map, vertical_half_variance = np.zeros(len(releases)), np.zeros(len(releases))
        front, shift = np.eye(2), np.zeros(2)
        p_front, q_front = np.zeros((2, 2)), np.zeros((2, 2))
        z_front, variance_front = 1., 0.
        first = max(0, np.searchsorted(self.edges, releases.min(), side='right') - 1)
        last = np.searchsorted(self.edges, end, side='left') - 1
        for index in range(last, first - 1, -1):
            upper = min(end, self.edges[index + 1])
            mask = (releases >= self.edges[index]) & (releases < upper)
            durations = np.r_[upper - releases[mask], upper - self.edges[index]]
            flow, translation, p_step, q_step, z_step, variance_step = general_step(
                self.gradient[index], self.velocity[index], durations, self.direction[index], self.order)
            transported = front @ flow
            shifted = shift + np.einsum('ij,nj->ni', front, translation)
            p = covariance_factor(np.concatenate((np.broadcast_to(p_front, p_step.shape), front @ p_step), axis=-1))
            q = covariance_factor(np.concatenate((np.broadcast_to(q_front, q_step.shape), front @ q_step), axis=-1))
            z = z_front * z_step
            variance = variance_front + z_front ** 2 * variance_step
            maps[mask], shifts[mask] = transported[:-1], shifted[:-1]
            parallel[mask], perpendicular[mask] = p[:-1], q[:-1]
            vertical_map[mask], vertical_half_variance[mask] = z[:-1], variance[:-1]
            front, shift, p_front, q_front = transported[-1], shifted[-1], p[-1], q[-1]
            z_front, variance_front = z[-1], variance[-1]
        values = maps, shifts, parallel, perpendicular, vertical_map, vertical_half_variance
        if not all(np.isfinite(value).all() for value in values):
            raise FloatingPointError('3D affine history cannot be represented; no clipping applied')
        return values
