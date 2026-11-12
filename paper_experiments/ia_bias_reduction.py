"""Second-order bias of an interior nonlinear least-squares estimator.

Research diagnostic only: this module does not move source estimates or alter
the fitting objective. For separable models, use derivatives of the full mean
and include free amplitudes as nuisance parameters, even when Q was eliminated.

Expanding the normal equations gives
b = (J'J)^-1 {sum_i H_i Cov(delta,r)_i - .5 J' tr(H_i Cov(delta))}.
For iid errors the first term vanishes (Box, 1971,
doi:10.1111/j.2517-6161.1971.tb00871.x). Correlated errors require both terms.
This local expansion does not correct deterministic transport-model bias or
justify using an unidentified or boundary estimate as an interior one.
"""

import numpy as np

from reduced_trust_region import ReducedResidual


def second_order_ls_bias(jacobian,hessian,noise_covariance,rcond=1e-10):
    jacobian=np.asarray(jacobian,float)
    hessian=np.asarray(hessian,float)
    if (jacobian.ndim!=2 or not jacobian.size
            or hessian.shape!=(jacobian.shape[0],jacobian.shape[1],jacobian.shape[1])
            or not np.isfinite(jacobian).all() or not np.isfinite(hessian).all()
            or not 0<rcond<1):
        raise ValueError('Finite mean Jacobian/Hessians and a valid rank cutoff required')
    if not np.allclose(hessian,hessian.swapaxes(1,2),rtol=1e-8,atol=1e-12):
        raise ValueError('Mean Hessians must be symmetric')
    norms=np.linalg.norm(jacobian,axis=0)
    if np.any(norms==0) or jacobian.shape[0]<jacobian.shape[1]:
        return dict(identified=False,bias=None,reason='Unidentified full mean derivative')
    scaled=jacobian/norms
    u,s,vt=np.linalg.svd(scaled,full_matrices=False)
    rank=int(np.sum(s>rcond*s[0]))
    if rank<len(norms):
        return dict(identified=False,bias=None,rank=rank,
                    reason='Do not use a pseudoinverse to fabricate a unique bias correction')
    inverse=(vt.T/s)@u.T
    if callable(noise_covariance):
        cov_times_inverse=np.asarray(noise_covariance(inverse.T),float)
    else:
        covariance=np.asarray(noise_covariance,float)
        if covariance.shape!=(len(jacobian),len(jacobian)) or not np.isfinite(covariance).all():
            raise ValueError('Noise covariance must match observation rows')
        cov_times_inverse=covariance@inverse.T
    if cov_times_inverse.shape!=inverse.T.shape or not np.isfinite(cov_times_inverse).all():
        raise ValueError('Covariance operator must preserve the supplied matrix shape')
    parameter_covariance=inverse@cov_times_inverse
    if not np.allclose(parameter_covariance,parameter_covariance.T,rtol=1e-8,atol=1e-12):
        raise ValueError('Projected noise covariance must be symmetric')
    parameter_covariance=(parameter_covariance+parameter_covariance.T)*.5
    if np.linalg.eigvalsh(parameter_covariance).min() < -1e-10*max(np.linalg.norm(parameter_covariance),1e-300):
        raise ValueError('Projected noise covariance must be positive semidefinite')
    # Cross-covariance of the first-order parameter displacement and residual.
    cross=cov_times_inverse.T-parameter_covariance@scaled.T
    hessian_scaled=hessian/norms[None,:,None]/norms[None,None,:]
    acceleration=np.einsum('nij,ij->n',hessian_scaled,parameter_covariance)
    correlation_term=np.einsum('nij,jn->i',hessian_scaled,cross)
    bias_scaled=(inverse@inverse.T)@correlation_term-.5*inverse@acceleration
    return dict(identified=True,rank=rank,bias=bias_scaled/norms,
        parameter_covariance=parameter_covariance/norms[:,None]/norms[None,:],
        scaled_jacobian_condition=float(s[0]/s[-1]),
        bias_correlation_component=((inverse@inverse.T)@correlation_term)/norms,
        note='Working second-order noise-bias approximation; no source correction or empirical gain claimed')


def separable_mean_derivatives(response,physical,coefficients,steps):
    """Differentiate the full mean G(theta)*a, retaining amplitude directions."""
    physical=np.asarray(physical,float)
    coefficients=np.asarray(coefficients,float)
    steps=np.asarray(steps,float)
    if steps.shape!=physical.shape or np.any(steps<=0):
        raise ValueError('One positive difference step per physical parameter required')
    matrix,derivative=response(physical,True)
    n,k,p=derivative.shape
    if matrix.shape!=(n,k) or p!=len(physical) or k!=len(coefficients):
        raise ValueError('Response and amplitude dimensions do not agree')
    jacobian=np.column_stack((np.einsum('nkd,k->nd',derivative,coefficients),matrix))
    hessian=np.zeros((n,p+k,p+k))
    for j,step in enumerate(steps):
        perturbation=np.zeros(p)
        perturbation[j]=step
        plus=response(physical+perturbation,True)[1]
        minus=response(physical-perturbation,True)[1]
        hessian[:,:p,j]=np.einsum('nkd,k->nd',(plus-minus)/(2*step),coefficients)
    hessian[:,:p,:p]=.5*(hessian[:,:p,:p]+hessian[:,:p,:p].swapaxes(1,2))
    hessian[:,:p,p:]=derivative.swapaxes(1,2)
    hessian[:,p:,:p]=derivative
    return matrix@coefficients,jacobian,hessian


def station_ar1_covariance_operator(times,stations,sigma,rho):
    """Apply independent-station AR1 covariance without a full N-by-N matrix."""
    times,stations=np.asarray(times,float),np.asarray(stations,int)
    sigma,rho=np.asarray(sigma,float),np.asarray(rho,float)
    if (times.ndim!=1 or stations.shape!=times.shape or not len(times)
            or sigma.ndim!=1 or rho.shape!=sigma.shape or not np.isfinite(times).all()
            or not np.isfinite(sigma).all() or np.any(sigma<0)
            or not np.isfinite(rho).all() or np.any(rho<0) or np.any(rho>=1)
            or stations.min()<0 or stations.max()>=len(sigma)):
        raise ValueError('Valid station-indexed stationary covariance required')
    blocks=[]
    for station in np.unique(stations):
        index=np.flatnonzero(stations==station)
        delta=np.abs(times[index,None]-times[None,index])
        blocks.append((index,sigma[station]**2*rho[station]**delta))
    def apply(values):
        values=np.asarray(values,float)
        if values.ndim!=2 or len(values)!=len(times):
            raise ValueError('Covariance operand must be an observation-by-column matrix')
        result=np.zeros_like(values)
        for index,covariance in blocks:
            result[index]=covariance@values[index]
        return result
    return apply


def correct_source_estimate(response,observed,physical,coefficients,lower,upper,
                            q_upper,noise_model,*,active_mask=None,relative_step=1e-4,
                            interior_sigma_margin=3.):
    """Working source-only correction, with NNLS Q refit at fixed new geometry.

    Reject active constraints rather than silently treating a boundary estimator
    as an interior one. Rejection keeps the original estimate; it is not an
    inversion gate or evidence that the original source is physically correct.
    """
    physical,coefficients,lower,upper=[np.asarray(v,float) for v in
        (physical,coefficients,lower,upper)]
    observed=np.asarray(observed,float)
    if active_mask is not None and np.any(np.asarray(active_mask)!=0):
        return dict(applied=False,reason='Active amplitude constraint: interior expansion is not applicable')
    if (np.any(physical<=lower) or np.any(physical>=upper)
            or np.any(coefficients<=0) or np.any(coefficients>=q_upper)):
        return dict(applied=False,reason='Active physical or amplitude bound')
    distance=np.minimum(physical-lower,upper-physical)
    steps=np.minimum(relative_step*np.maximum(np.abs(physical),.01),distance*.25)
    prediction,jacobian,hessian=separable_mean_derivatives(response,physical,coefficients,steps)
    covariance,noise_record=noise_model(prediction-observed,jacobian)
    coarse=second_order_ls_bias(jacobian,hessian,covariance)
    if not coarse['identified']:
        return dict(applied=False,reason=coarse['reason'],noise_model=noise_record)
    _,_,fine_hessian=separable_mean_derivatives(response,physical,coefficients,steps*.5)
    fine=second_order_ls_bias(jacobian,fine_hessian,covariance)
    change=float(np.linalg.norm(fine['bias'][:2]-coarse['bias'][:2])/
        max(np.linalg.norm(fine['bias'][:2]),np.finfo(float).eps))
    record=dict(bias_analysis=fine,noise_model=noise_record,finite_difference_relative_change=change,
        original_sse=float(np.sum((prediction-observed)**2)),steps=steps)
    if not np.isfinite(fine['bias']).all() or change>1e-3:
        return dict(applied=False,reason='Numerical bias derivative check failed',**record)
    variance=np.diag(fine['parameter_covariance'])
    if np.any(variance<0):
        return dict(applied=False,reason='Numerically invalid parameter variance',**record)
    deviation=np.sqrt(variance)
    parameter=np.r_[physical,coefficients]
    all_lower=np.r_[lower,np.zeros(len(coefficients))]
    all_upper=np.r_[upper,np.broadcast_to(q_upper,coefficients.shape)]
    distance=np.minimum(parameter-all_lower,all_upper-parameter)
    margin=np.divide(distance,deviation,out=np.full_like(distance,np.inf),where=deviation>0)
    ratio=np.divide(np.abs(fine['bias']),deviation,out=np.zeros_like(deviation),where=deviation>0)
    record.update(minimum_boundary_sigma_margin=float(margin.min()) if np.isfinite(margin.min()) else None,
                  maximum_bias_to_standard_error=float(ratio.max()))
    if np.any(margin<interior_sigma_margin):
        return dict(applied=False,reason='Noise neighborhood reaches a bound; interior bias expansion is not applicable',**record)
    if np.any(ratio>1.):
        return dict(applied=False,reason='Second-order bias is not small relative to first-order uncertainty',**record)
    corrected=physical.copy()
    corrected[:2]-=fine['bias'][:2]
    if np.any(corrected[:2]<lower[:2]) or np.any(corrected[:2]>upper[:2]):
        return dict(applied=False,reason='Corrected location leaves original search bounds',**record)
    # Re-estimate only Q. There is no location or diffusion optimization here.
    solver=ReducedResidual(response,observed,q_upper_bound=q_upper)
    solver.residual(corrected)
    return dict(applied=True,physical=corrected,coefficients=solver.solution['coefficients'],
        prediction=solver.solution['prediction'],sse=float(solver.residual_value@solver.residual_value),
        note='Only source coordinates are bias-corrected; nuisance D unchanged and Q re-profiled',**record)
