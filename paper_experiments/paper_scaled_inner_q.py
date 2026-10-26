"""Numerically scaled nonnegative Green-response least squares for experiments.

z_j = ||WA_j|| a_j is a change of coordinates, not a prior or a new loss.
Only exactly zero weighted columns are removed; weak columns are not thresholded.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pinn_source'))
from variable_projection import NonnegativeQSolution,timed_component


def as_numpy(value):
    return np.asarray(value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value,dtype=float)


@timed_component('inner_q_solve')
def solve_scaled_nonnegative_q(response_matrix,observations,weights=None,coefficient_upper_bound=np.inf):
    response=as_numpy(response_matrix)
    observed=as_numpy(observations).reshape(-1)
    if (response.ndim!=2 or response.shape[0]!=len(observed) or
            not np.isfinite(response).all() or not np.isfinite(observed).all() or np.any(response<0)):
        raise ValueError('Finite nonnegative Green response and matching observations required')
    weight=np.ones(len(observed)) if weights is None else as_numpy(weights).reshape(-1)
    if weight.shape!=observed.shape or not np.isfinite(weight).all() or np.any(weight<0):
        raise ValueError('Finite nonnegative residual weights required')
    upper=float(coefficient_upper_bound)
    if np.isnan(upper) or upper<=0:
        raise ValueError('Positive coefficient upper bound or infinity required')
    a=response*weight[:,None]
    b=observed*weight
    scales=np.linalg.norm(a,axis=0)
    nonzero=scales>0
    coefficients=np.zeros(response.shape[1])
    if nonzero.any():
        normalized=a[:,nonzero]/scales[nonzero]
        bounds=scales[nonzero]*upper if np.isfinite(upper) else np.full(nonzero.sum(),np.inf)
        if np.any(bounds==0):
            raise FloatingPointError('Scaled coefficient bound underflow')
        fit=lsq_linear(normalized,b,bounds=(np.zeros(nonzero.sum()),bounds),
                       method='bvls',tol=1e-12,max_iter=100)
        coefficients[nonzero]=fit.x/scales[nonzero]
        if not np.isfinite(coefficients).all():
            raise FloatingPointError('Effective source coefficient exceeds float64 range')
        gradient=normalized.T@(normalized@fit.x-b)
        projected=fit.x-np.clip(fit.x-gradient,0.,bounds)
        kkt=float(np.max(np.abs(projected))/max(1.,np.linalg.norm(b)))
        success=bool(fit.success and kkt<=1e-9)
        status=int(fit.status) if success else 0
        message=f'Column-scaled SciPy BVLS: {fit.message}; normalized projected KKT={kkt:.3g}'
    else:
        success,status,message=True,3,'Exactly zero weighted response; canonical zero coefficients'
    prediction=response@coefficients
    residual=prediction-observed
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite reconstructed prediction')
    singular=np.linalg.svd(a,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular.size and singular[-1]>np.finfo(float).eps*singular[0] else float('inf')
    return NonnegativeQSolution(coefficients=coefficients,prediction=prediction,residual=residual,
        weighted_sse=float(np.sum((weight*residual)**2)),success=success,status=status,message=message,
        response_rank=int(np.linalg.matrix_rank(a)),response_condition=condition)
