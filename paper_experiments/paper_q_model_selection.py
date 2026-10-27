"""Research-only nested release-capacity selection, with pure SSE fits per model.

BIC is a Gaussian working-likelihood criterion, not a new residual penalty.
Its regular-model interpretation is approximate at bounds or unidentified sources.
"""

import time

import numpy as np

from paper_joint_baselines import JointBounds, JointResidual
from paper_reduced_solver import fit_profiled_trf
from q_capacity import nested_hat_transform, gaussian_bic


def fit_profiled_q_models(objective, starts, counts=(1, 3, 5), max_nfev=250, solver='profiled'):
    begun = time.perf_counter()
    base_count = objective.basis.shape[1]
    if len(set(counts)) != len(counts) or not counts:
        raise ValueError('Distinct nonempty model capacities required')
    models = []
    for count in sorted(counts):
        transform = nested_hat_transform(base_count, count)
        transfer = objective.tensor(transform)
        small_basis = objective.tensor(objective.basis) @ transfer

        def response(model, sigma, xyt, basis, **kwargs):
            return objective.response(model, sigma, xyt, objective.basis, **kwargs) @ transfer

        if hasattr(objective.response,'matrix_and_jacobian'):
            def matrix_and_jacobian(model,sigma,xyt,basis,physical_parameters):
                matrix,derivative=objective.response.matrix_and_jacobian(model,sigma,xyt,
                    objective.basis,physical_parameters=physical_parameters)
                import torch
                return matrix@transfer,torch.einsum('nkd,kj->njd',derivative,transfer)
            response.matrix_and_jacobian=matrix_and_jacobian

        cap = objective.bounds.upper[4:]
        if not np.all(cap == cap[0]):
            raise ValueError('Common amplitude bound required across nested models')
        bounds = JointBounds(np.r_[objective.bounds.lower[:4], np.zeros(count)],
                             np.r_[objective.bounds.upper[:4], np.full(count, cap[0])])
        small = JointResidual(response, objective.model, objective.sigma,
            objective.xyt, small_basis, objective.observations, bounds)
        # Q is eliminated; its initial coordinates do not influence the outer start.
        initial = [np.r_[np.asarray(value)[:4], np.zeros(count)] for value in starts]
        if solver == 'profiled':
            fit = fit_profiled_trf(small, initial, max_nfev)
        elif solver == 'joint_nls':
            from paper_beam_joint_baselines import BeamJointResidual
            from paper_joint_baselines import fit_joint_nls
            from reduced_trust_region import ReducedResidual
            joint = BeamJointResidual(response, objective.model, objective.sigma,
                objective.xyt, small_basis, objective.observations, bounds)
            initializer = ReducedResidual(lambda theta, unused: response(objective.model,
                objective.sigma, objective.xyt, small_basis,
                physical_parameters=objective.tensor(theta)).detach().cpu().numpy(),
                objective.observations.detach().cpu().numpy(), q_upper_bound=float(cap[0]))
            for value in initial:
                initializer.residual(value[:4])
                value[4:] = initializer.solution['coefficients']
            fit = fit_joint_nls(joint, initial, max_nfev)
        else:
            raise ValueError('Unknown release-capacity solver')
        fit['native_coefficients'] = np.asarray(fit['coefficients']).copy()
        fit['coefficients'] = transform @ fit['native_coefficients']
        fit['q_basis_count'] = int(count)
        fit['bic'] = gaussian_bic(fit['sse'], objective.observations.numel(), 4+count)
        models.append(fit)
    selected = min(models, key=lambda item: (item['bic'], item['q_basis_count']))
    result = dict(selected)
    result.update(method=('ia_vpdi_release_bic' if solver == 'profiled' else 'joint_nls_release_bic'),
        seconds=time.perf_counter()-begun,
        selection='BIC: n*log(SSE/n)+(4+Q_count)*log(n); all observations; no truth',
        selection_note='Working Gaussian model; asymptotic evidence not certified at bounds/rank loss',
        release_models=[{k: item[k] for k in ('q_basis_count', 'bic', 'sse',
            'source_norm', 'native_coefficients', 'converged', 'seconds')} for item in models])
    return result
