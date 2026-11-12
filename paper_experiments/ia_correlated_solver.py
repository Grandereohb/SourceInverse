"""Five-coordinate IA variable projection for correlated-velocity transport."""

import numpy as np
import torch
from q_capacity import nested_hat_transform
from reduced_trust_region import fit_reduced_trust_region


def response_factory(model, xyt, basis, response, count):
    mapping = nested_hat_transform(basis.shape[1], count)
    transfer = model.xs.new_tensor(mapping)
    def evaluate(theta, jacobian):
        parameters = model.xs.new_tensor(theta)
        if jacobian:
            matrix, derivative = response.matrix_and_jacobian(model, 0., xyt, basis, parameters)
            derivative = torch.einsum('nkd,kj->njd', derivative, transfer)
            return (matrix @ transfer).cpu().numpy(), derivative.cpu().numpy()
        return (response(model, 0., xyt, basis, physical_parameters=parameters) @ transfer).cpu().numpy()
    return evaluate, mapping


def correlation_information(matrix, derivative, coefficients, active_mask):
    """Local tangent diagnostic, not a calibrated confidence interval or gate."""
    physical = np.einsum('nkd,k->nd', derivative, coefficients)
    nuisance = np.column_stack((physical[:, :4], matrix[:, np.asarray(active_mask) == 0]))
    scale = np.linalg.norm(nuisance, axis=0)
    normalized = nuisance[:, scale > 0] / scale[scale > 0]
    u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    valid = singular > 1e-10 * singular[0] if len(singular) else np.zeros(0, bool)
    tau = physical[:, 4]
    remaining = tau - u[:, valid] @ (u[:, valid].T @ tau)
    norm = np.linalg.norm(tau)
    return dict(tau_derivative_norm=float(norm),
        tau_remaining_fraction=float(np.linalg.norm(remaining) / norm) if norm > 0 else None,
        nuisance_rank=int(valid.sum()),
        interpretation='Unrestricted local nuisance tangent, including D at bounds; no confidence calibration')


def fit_correlated(model, xyt, basis, response, observed, starts, lower, upper,
                   q_upper, count, max_nfev=100):
    function, mapping = response_factory(model, xyt, basis, response, count)
    fit = fit_reduced_trust_region(function, observed, starts, lower, upper,
        q_upper_bound=q_upper, max_nfev=max_nfev, coordinate_scale=np.array([1., 1., .03, .03, 1.]))
    theta = fit['theta']
    matrix, derivative = function(theta, True)
    fit['correlation_information'] = correlation_information(matrix, derivative,
        fit['coefficients'], fit['active_mask'])
    native = np.asarray(fit['coefficients']).copy()
    fit.update(source_norm=theta[:2], diffusion_norm=[theta[2] + theta[3], theta[2]],
        correlation_time_s=float(theta[4]), native_coefficients=native,
        coefficients=mapping @ native, q_basis_count=count, mse=fit['sse'] / len(observed),
        nls_called=False, method='ia_correlated_velocity_profiled')
    return fit
