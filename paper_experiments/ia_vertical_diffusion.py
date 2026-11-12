"""IA-only independent vertical diffusivity with profiled nonnegative emissions."""

import time
import numpy as np
import torch
from q_capacity import nested_hat_transform,gaussian_bic
from reduced_trust_region import fit_reduced_trust_region


def vertical_response(objective,count):
    mapping=nested_hat_transform(objective.basis.shape[1],count)
    transfer=objective.tensor(mapping)
    def response(theta,jacobian):
        value,derivative=objective.response.matrix_and_jacobian(objective.model,
            objective.sigma,objective.xyt,objective.basis,
            physical_parameters=objective.tensor(theta))
        value=value@transfer
        if jacobian:
            derivative=torch.einsum('nkd,kj->njd',derivative,transfer)
            return value.cpu().numpy(),derivative.cpu().numpy()
        return value.cpu().numpy()
    return response,mapping


def fit_vertical_diffusion(objective,starts,reference,max_nfev=250):
    begun=time.perf_counter()
    count=reference['q_basis_count']
    response,mapping=vertical_response(objective,count)
    base=np.r_[reference['source_norm'],reference['diffusion_norm'][1],
               reference['diffusion_norm'][0]-reference['diffusion_norm'][1]]
    initial=[np.r_[np.asarray(start)[:4],np.asarray(start)[2]] for start in starts]
    initial.append(np.r_[base,base[2]])
    bounds=objective.bounds
    lower=np.r_[bounds.lower[:4],bounds.lower[2]]
    upper=np.r_[bounds.upper[:4],bounds.upper[2]]
    observed=objective.observations.cpu().numpy()
    fit=fit_reduced_trust_region(response,observed,initial,lower,upper,
        q_upper_bound=float(bounds.upper[4]),max_nfev=max_nfev)
    theta=fit.pop('theta')
    native=fit['coefficients'].copy()
    fit.update(source_norm=theta[:2],diffusion_norm=[theta[2]+theta[3],theta[2]],
        vertical_diffusion_norm=float(theta[4]),native_coefficients=native,
        coefficients=mapping@native,q_basis_count=count,parameter_count=5+count,
        bic=gaussian_bic(fit['sse'],len(observed),5+count),seconds=time.perf_counter()-begun)
    original=dict(reference)
    matrix=response(np.r_[base,base[2]],False)
    original['prediction']=matrix@np.asarray(reference['native_coefficients'])
    original['sse']=float(np.sum((original['prediction']-observed)**2))
    original['bic']=gaussian_bic(original['sse'],len(observed),4+count)
    chosen=fit if fit['bic']<original['bic'] else original
    return dict(selected_mode='independent_vertical' if chosen is fit else 'tied_vertical',
        selected=chosen,independent=fit,tied=original,
        selection='Working Gaussian BIC, same Q capacity and observations, no source truth')
