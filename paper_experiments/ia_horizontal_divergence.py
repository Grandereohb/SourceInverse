"""Wind-only diagnostic; a horizontal divergence is not a measured vertical wind."""

import numpy as np


def fit_horizontal_affine(positions, velocities):
    positions = np.asarray(positions, float)
    velocities = np.asarray(velocities, float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or velocities.ndim != 3 or
            velocities.shape[1:] != positions.shape or
            not np.isfinite(positions).all() or not np.isfinite(velocities).all()):
        raise ValueError('Finite sensor positions and time-by-sensor ENU vectors required')
    center = positions.mean(0)
    length = np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1)))
    if length == 0:
        raise ValueError('Spatially distinct wind sensors required')
    design = np.column_stack((np.ones(len(positions)), (positions - center) / length))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError('Noncollinear wind sensors required')
    coefficients = np.einsum('ks,tsd->tkd', np.linalg.pinv(design), velocities)
    gradient = coefficients[:, 1:].transpose(0, 2, 1) / length
    intercept = coefficients[:, 0] - np.einsum('tij,j->ti', gradient, center)
    fitted = intercept[:, None] + np.einsum('tij,sj->tsi', gradient, positions)
    return dict(intercept=intercept, gradient=gradient, fitted=fitted,
        divergence=np.trace(gradient, axis1=1, axis2=2),
        design_condition=float(np.linalg.cond(design)),
        residual_degrees_of_freedom_per_time=2 * len(positions) - 6,
        caveat='Three wind sensors exactly identify six coefficients, leaving zero residual degrees of freedom; nonzero divergence is not independently validated')


def incompressible_3d_gradient(horizontal):
    """Kinematic completion w=-tr(Ah)*z, not a claim that this closure is true."""
    horizontal = np.asarray(horizontal, float)
    result = np.zeros((*horizontal.shape[:-2], 3, 3))
    result[..., :2, :2] = horizontal
    result[..., 2, 2] = -np.trace(horizontal, axis1=-2, axis2=-1)
    return result
