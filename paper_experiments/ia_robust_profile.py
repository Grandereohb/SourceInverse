"""Research-only Huber variable projection; frozen NLS is not imported.

Replace, rather than augment, the observation likelihood. Amplitudes remain a
convex nonnegative inner problem and all observations retain nonzero influence.
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize


def huber_terms(residual, delta=1.345):
    residual = np.asarray(residual, float)
    absolute = np.abs(residual)
    return (np.where(absolute <= delta, .5 * residual**2,
                     delta * (absolute - .5 * delta)),
            np.clip(residual, -delta, delta))


def solve_huber_amplitudes(matrix, observed, scale, upper, delta=1.345):
    matrix, observed = np.asarray(matrix, float), np.asarray(observed, float)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('A positive fixed observation scale is required')
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > 0
    coefficients = np.zeros(matrix.shape[1])
    upper = np.broadcast_to(upper, coefficients.shape)
    if np.any(upper <= 0):
        raise ValueError('Amplitude upper bounds must be positive')
    if not np.any(active):
        return dict(coefficients=coefficients, converged=True, projected_gradient=0.)
    design = matrix[:, active] / norms[active]
    target = observed / scale
    limit = upper[active] * norms[active] / scale
    initial = lsq_linear(design, target, bounds=(0., limit), method='bvls',
                         tol=1e-12, max_iter=500).x

    def objective(beta):
        rho, psi = huber_terms(design @ beta - target, delta)
        return float(rho.sum()), design.T @ psi

    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=list(zip(np.zeros_like(limit), limit)),
                      options=dict(maxiter=500, ftol=1e-14, gtol=1e-9, maxls=40))
    gradient = objective(result.x)[1]
    projected = result.x - np.clip(result.x - gradient, 0., limit)
    kkt = float(np.max(np.abs(projected)))
    coefficients[active] = result.x * scale / norms[active]
    return dict(coefficients=coefficients, converged=kkt < 1e-5,
                projected_gradient=kkt, message=str(result.message))


def profiled_value_gradient(response, theta, observed, scale, upper, delta=1.345):
    matrix, derivative = response(theta, True)
    inner = solve_huber_amplitudes(matrix, observed, scale, upper, delta)
    coefficients = inner['coefficients']
    residual = matrix @ coefficients - observed
    rho, psi = huber_terms(residual / scale, delta)
    # Envelope derivative: the optimized amplitudes need not be differentiated.
    physical = np.einsum('nkd,k->nd', derivative, coefficients)
    gradient = physical.T @ psi / scale
    return float(rho.sum()), gradient, inner


def fit_robust_profile(response, observed, starts, lower, upper, q_upper,
                       scale, maxiter=250, delta=1.345):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    if np.any(width <= 0):
        raise ValueError('Physical coordinates must have nonzero bounded ranges')
    trials = []
    for start in starts:
        failures = 0

        def objective(unit):
            nonlocal failures
            value, gradient, inner = profiled_value_gradient(
                response, lower + unit * width, observed, scale, q_upper, delta)
            failures += not inner['converged']
            return value / len(observed), gradient * width / len(observed)

        result = minimize(objective, np.clip((start-lower)/width, 0., 1.),
                          jac=True, bounds=[(0., 1.)] * len(lower), method='L-BFGS-B',
                          options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxls=30))
        theta = lower + result.x * width
        value, gradient, inner = profiled_value_gradient(
            response, theta, observed, scale, q_upper, delta)
        matrix = response(theta, False)
        robust_prediction = matrix @ inner['coefficients']
        # Mean amplitude is refitted at the selected geometry; it cannot move it.
        norms = np.linalg.norm(matrix, axis=0)
        valid = norms > 0
        mean_coefficients = np.zeros(matrix.shape[1])
        if np.any(valid):
            mean_coefficients[valid] = lsq_linear(
                matrix[:, valid]/norms[valid], observed,
                bounds=(0., np.broadcast_to(q_upper, norms.shape)[valid]*norms[valid]),
                method='bvls', tol=1e-12, max_iter=500).x / norms[valid]
        trials.append(dict(theta=theta, huber_objective=value, noise_scale=scale,
            delta=delta, coefficients=mean_coefficients,
            robust_coefficients=inner['coefficients'], prediction=matrix@mean_coefficients,
            robust_prediction=robust_prediction, converged=bool(result.success and inner['converged']),
            inner_kkt=inner['projected_gradient'], inner_nonconverged_calls=int(failures),
            nfev=int(result.nfev), message=str(result.message)))
    valid = [trial for trial in trials if trial['inner_kkt'] < 1e-5]
    if not valid:
        raise RuntimeError('No IA candidate has an adequately solved convex inner problem')
    best = min(valid, key=lambda trial: trial['huber_objective'])
    best['trials'] = [{k:v for k,v in trial.items() if k not in
                       ('prediction','robust_prediction')} for trial in trials]
    return best
