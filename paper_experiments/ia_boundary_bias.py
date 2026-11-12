"""IA-only working boundary-bias estimate from a local bounded Gaussian model.

For a full-rank mean Jacobian J and measurement covariance C, unconstrained
linearized displacements have covariance J+ C J+T. Project their Gaussian
draws in the J.T J metric onto the original parameter box, then integrate
the displacement. This is a finite-box plug-in approximation, not a calibrated
boundary bootstrap, confidence interval, or guarantee of nonlinear bias removal.
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import ndtri
from scipy.stats import qmc
from ia_bias_reduction import second_order_ls_bias
from reduced_trust_region import ReducedResidual


def linearized_box_bias(jacobian,covariance,center,lower,upper,*,samples=4096,seed=730001):
    jacobian=np.asarray(jacobian,float)
    center,lower,upper=[np.asarray(v,float) for v in (center,lower,upper)]
    p=len(center)
    if (jacobian.ndim!=2 or jacobian.shape[1]!=p or lower.shape!=center.shape
            or upper.shape!=center.shape or np.any(lower>=upper)
            or np.any(center<lower) or np.any(center>upper)
            or samples<8 or samples&(samples-1)):
        raise ValueError('Matching feasible box and power-of-two Gaussian sample count required')
    moments=second_order_ls_bias(jacobian,np.zeros((len(jacobian),p,p)),covariance)
    if not moments['identified']:
        return dict(available=False,reason=moments['reason'])
    scale=np.linalg.norm(jacobian,axis=0)
    _,singular,vt=np.linalg.svd(jacobian/scale,full_matrices=False)
    metric=singular[:,None]*vt
    v=moments['parameter_covariance']*scale[:,None]*scale[None,:]
    if np.all(v==0):
        return dict(available=True,bias=np.zeros(p),coarse_bias=np.zeros(p),projected_fraction=0.,
            integration_relative_change=0.,samples=samples,parameter_covariance=moments['parameter_covariance'])
    try:
        factor=np.linalg.cholesky(v)
    except np.linalg.LinAlgError:
        return dict(available=False,reason='Projected covariance is not numerically positive definite; no eigenvalue repair')
    half=samples//2
    gaussian=ndtri(qmc.Sobol(d=p,scramble=True,seed=seed).random_base2(int(np.log2(half))))
    unconstrained=np.vstack((gaussian,-gaussian))@factor.T
    low=(lower-center)*scale
    high=(upper-center)*scale
    projected=unconstrained.copy()
    changed=0
    worst_optimality=0.
    for i,draw in enumerate(unconstrained):
        if np.all(draw>=low) and np.all(draw<=high):
            continue
        solution=lsq_linear(metric,metric@draw,bounds=(low,high),method='bvls',tol=1e-11,max_iter=500)
        if not solution.success:
            raise RuntimeError('Local bounded Gaussian projection failed: '+str(solution.message))
        projected[i]=solution.x
        changed+=1
        worst_optimality=max(worst_optimality,float(solution.optimality))
    # Both estimates use paired +/- draws, so interior Gaussian means cancel.
    paired=.5*(projected[:half]+projected[half:])/scale
    bias=paired.mean(0)
    coarse=paired[:half//2].mean(0)
    relative=float(np.linalg.norm(bias[:2]-coarse[:2])/max(np.linalg.norm(bias[:2]),1e-15))
    return dict(available=True,bias=bias,coarse_bias=coarse,
        parameter_covariance=moments['parameter_covariance'],
        integration_relative_change=relative,projected_fraction=changed/samples,samples=samples,
        worst_linear_optimality=worst_optimality,scaled_jacobian_condition=moments['scaled_jacobian_condition'],
        note='Local finite-box Gaussian plug-in; not a consistent boundary-bootstrap claim')


def correct_boundary_source(response,observed,physical,coefficients,lower,upper,q_upper,noise_model,
                            *,samples=4096,seed=730001):
    physical,coefficients,lower,upper=[np.asarray(v,float) for v in (physical,coefficients,lower,upper)]
    observed=np.asarray(observed,float)
    matrix,derivative=response(physical,True)
    mean=matrix@coefficients
    jacobian=np.column_stack((np.einsum('nkd,k->nd',derivative,coefficients),matrix))
    covariance,noise_record=noise_model(mean-observed,jacobian)
    result=linearized_box_bias(jacobian,covariance,np.r_[physical,coefficients],
        np.r_[lower,np.zeros(len(coefficients))],
        np.r_[upper,np.broadcast_to(q_upper,coefficients.shape)],samples=samples,seed=seed)
    record=dict(boundary_analysis=result,noise_model=noise_record,
        original_sse=float(np.sum((mean-observed)**2)))
    if not result['available']:
        return dict(applied=False,reason=result['reason'],**record)
    if result['integration_relative_change']>.05:
        return dict(applied=False,reason='Local Gaussian integration has not stabilized at the fixed budget',**record)
    corrected=physical.copy()
    corrected[:2]-=result['bias'][:2]
    if np.any(corrected[:2]<lower[:2]) or np.any(corrected[:2]>upper[:2]):
        return dict(applied=False,reason='Bias-corrected source leaves original search box',**record)
    new_matrix=response(corrected,False)
    actual_shift=new_matrix@coefficients-mean
    linear_shift=jacobian[:,:len(physical)]@(corrected-physical)
    remainder=float(np.linalg.norm(actual_shift-linear_shift)/max(np.linalg.norm(linear_shift),1e-14*max(np.linalg.norm(mean),1.)))
    record['relative_mean_linearization_remainder']=remainder
    if remainder>.1:
        return dict(applied=False,reason='Exact forward check rejects the local mean approximation',**record)
    solver=ReducedResidual(response,observed,q_upper_bound=q_upper)
    solver.residual(corrected)
    return dict(applied=True,physical=corrected,source_norm=corrected[:2],
        coefficients=solver.solution['coefficients'],prediction=solver.solution['prediction'],
        sse=float(solver.residual_value@solver.residual_value),**record)
