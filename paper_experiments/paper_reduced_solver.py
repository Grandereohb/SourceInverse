"""Physical-response adapter for scaled, active-set variable-projection TRF."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pinn_source'))
from reduced_trust_region import fit_reduced_trust_region


def fit_profiled_trf(objective, starts, max_nfev=250):
    if not starts:
        raise ValueError('Common observation-only starts required')
    cap = objective.bounds.upper[4:]
    if not np.all(cap == cap[0]) or np.any(objective.bounds.lower[4:] != 0):
        raise ValueError('Shared nonnegative Q bounds required')

    def response(theta, jacobian):
        values = objective.tensor(theta)
        def calculate(parameters):
            return objective.response(objective.model, objective.sigma,
                objective.xyt, objective.basis, track_gradients=True,
                physical_parameters=parameters)
        if jacobian:
            if hasattr(objective.response,'matrix_and_jacobian'):
                matrix,derivative=objective.response.matrix_and_jacobian(objective.model,
                    objective.sigma,objective.xyt,objective.basis,physical_parameters=values)
                return matrix.cpu().numpy(),derivative.cpu().numpy()
            with torch.enable_grad():
                matrix = calculate(values)
                derivative = torch.autograd.functional.jacobian(
                    calculate, values, vectorize=True, strategy='forward-mode')
            return matrix.detach().cpu().numpy(), derivative.detach().cpu().numpy()
        with torch.no_grad():
            return calculate(values).detach().cpu().numpy()

    fit = fit_reduced_trust_region(response, objective.observations.detach().cpu().numpy(),
        [value[:4] for value in starts], objective.bounds.lower[:4], objective.bounds.upper[:4],
        q_upper_bound=float(cap[0]), max_nfev=max_nfev)
    theta = fit.pop('theta')
    fit.update(source_norm=theta[:2].tolist(),
        diffusion_norm=[float(theta[2]+theta[3]), float(theta[2])],
        mse=fit['sse']/objective.observations.numel(), method='ia_vpdi_reduced_trf')
    return fit


def standard_pso_scales(starts, lower, upper):
    """Fixed scales from the declared domain and common starts, never source truth."""
    starts = np.asarray(starts, dtype=float)
    scale = np.maximum(np.max(np.abs(starts), axis=0), 1e-3)
    scale[:2] = .25*(np.asarray(upper)[:2]-np.asarray(lower)[:2])
    scale[4:] = max(float(np.max(starts[:, 4:])), 1e-6)
    return scale
