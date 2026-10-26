"""Independent NumPy Gaussian-puff reference for analytic synthetic winds.

This is a two-dimensional, unbounded, finite-width mass-normalized source.
It is not a reproduction of an external paper and does not define kg/h for
the production two-dimensional concentration model.
"""

import numpy as np
from scipy.special import roots_legendre


def puff_response(stations, times, knots, source, d_parallel, d_perp,
                  *, wind='steady', speed=1.2, sigma=0.15, decay=0.1, order=32,
                  angle_offset=0.0):
    """Return (time,station,basis) responses with exact wind/tensor integrals.

    Release-time integration uses Gauss-Legendre quadrature within every hat
    interval. The wind angle is0.05 for steady wind, or -0.35+0.15*t radians.
    No production transport, interpolation, basis, model or solver code is used.
    """
    stations, times, knots, source = [np.asarray(x, dtype=float) for x in (stations, times, knots, source)]
    scalars = np.asarray([d_parallel, d_perp, speed, sigma, decay, angle_offset], dtype=float)
    if (stations.ndim != 2 or stations.shape[1] != 2 or source.shape != (2,)
            or times.ndim != 1 or knots.ndim != 1 or len(knots) < 2
            or not all(np.isfinite(x).all() for x in (stations, times, knots, source, scalars))
            or not np.all(np.diff(knots) > 0) or knots[0] != 0
            or np.any(times < 0) or np.any(times > knots[-1])
            or not d_parallel >= d_perp > 0 or sigma <= 0 or min(speed, decay) < 0
            or int(order) != order or order < 2 or wind not in ('steady', 'veering')):
        raise ValueError('Invalid finite source, transport, time or quadrature specification')
    nodes, weights = roots_legendre(int(order))
    output = np.zeros((len(times), len(stations), len(knots)))
    for i, end in enumerate(times):
        if end == 0:
            continue
        breaks = np.r_[knots[knots < end], end]
        left, right = breaks[:-1], breaks[1:]
        releases = ((left[:, None]+right[:, None])*0.5 +
                    (right-left)[:, None]*0.5*nodes).ravel()
        integration_weights = ((right-left)[:, None]*0.5*weights).ravel()
        age = end-releases
        contrast = d_parallel-d_perp
        if wind == 'steady':
            angle = 0.05+angle_offset
            c, s = np.cos(angle), np.sin(angle)
            displacement_x, displacement_y = speed*c*age, speed*s*age
            integral_xx = (d_perp+contrast*c*c)*age
            integral_yy = (d_perp+contrast*s*s)*age
            integral_xy = contrast*c*s*age
        else:
            angular_rate = 0.15
            release_angle, end_angle = -0.35+angle_offset+angular_rate*releases, -0.35+angle_offset+angular_rate*end
            displacement_x = speed/angular_rate*(np.sin(end_angle)-np.sin(release_angle))
            displacement_y = speed/angular_rate*(np.cos(release_angle)-np.cos(end_angle))
            anisotropy = contrast/(4*angular_rate)*(np.sin(2*end_angle)-np.sin(2*release_angle))
            integral_xx = (d_parallel+d_perp)*0.5*age+anisotropy
            integral_yy = (d_parallel+d_perp)*0.5*age-anisotropy
            integral_xy = contrast/(4*angular_rate)*(np.cos(2*release_angle)-np.cos(2*end_angle))
        cxx, cyy, cxy = sigma*sigma+2*integral_xx, sigma*sigma+2*integral_yy, 2*integral_xy
        determinant = cxx*cyy-cxy*cxy
        if np.any(determinant <= 0):
            raise FloatingPointError('Nonpositive integrated covariance determinant')
        dx = stations[:, 0, None]-source[0]-displacement_x
        dy = stations[:, 1, None]-source[1]-displacement_y
        exponent = (cyy*dx*dx+cxx*dy*dy-2*cxy*dx*dy)/determinant
        kernels = np.exp(-0.5*exponent)/(2*np.pi*np.sqrt(determinant))
        basis = np.stack([np.interp(releases, knots, np.eye(len(knots))[j]) for j in range(len(knots))], axis=1)
        output[i] = (kernels*(integration_weights*np.exp(-decay*age))) @ basis
    return output
