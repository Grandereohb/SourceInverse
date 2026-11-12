"""Location-oriented release-capacity selection with unchanged NNLS/TRF fits.

All observations enter every fit. A working Gaussian nested-model compatibility
check limits mean-fit bias; among compatible fits, minimize local source
variance after eliminating diffusion and free amplitudes. This is a local
working criterion, not a calibrated confidence region under model mismatch.
"""

import time

import numpy as np
from scipy.stats import chi2
import torch

from q_capacity import nested_hat_transform
from reduced_trust_region import fit_reduced_trust_region


def location_variance(jacobian,noise_variance,length_m,rcond=1e-10):
    matrix=np.asarray(jacobian,float)
    source=matrix[:,:2]/length_m
    reference=np.linalg.norm(source,ord=2)
    nuisance=matrix[:,2:]
    scale=np.linalg.norm(nuisance,axis=0)
    nuisance=nuisance[:,scale>0]/scale[scale>0]
    if nuisance.shape[1]:
        u,s,_=np.linalg.svd(nuisance,full_matrices=False)
        valid=s>rcond*s[0]
        source=source-u[:,valid]@(u[:,valid].T@source)
    u,s,vt=np.linalg.svd(source,full_matrices=False)
    valid=s>rcond*reference if s.size and reference>0 else np.zeros_like(s,dtype=bool)
    rank=int(valid.sum())
    if rank<2:
        return dict(source_rank=rank,variance_trace_m2=None,covariance_m2=None,
            warning='Source direction is unidentifiable; do not replace missing variance by zero')
    covariance=noise_variance*(vt.T/(s*s))@vt
    return dict(source_rank=2,variance_trace_m2=float(np.trace(covariance)),
        covariance_m2=covariance,source_information_condition=float((s[0]/s[-1])**2))


def choose_location_capacity(models,n_observations,*,alpha=.05):
    best=min(models,key=lambda fit:fit['sse'])
    maximum=max(fit['q_basis_count'] for fit in models)
    variance=max(best['sse']/max(n_observations-(4+maximum),1),np.finfo(float).tiny)
    for fit in models:
        delta=maximum-fit['q_basis_count']
        threshold=float(chi2.ppf(1-alpha,delta)*variance) if delta else 0.
        fit['fit_excess_sse']=max(0.,float(fit['sse']-best['sse']))
        fit['fit_compatibility_threshold']=threshold
        fit['fit_compatible']=bool(fit['fit_excess_sse']<=threshold+1e-8*max(best['sse'],1e-30))
    compatible=[fit for fit in models if fit['fit_compatible'] and fit['location_information']['variance_trace_m2'] is not None]
    selected=min(compatible,key=lambda fit:(fit['location_information']['variance_trace_m2'],fit['q_basis_count'])) if compatible else best
    return selected,variance,not bool(compatible)


def capacity_response(objective,count,physical_dimension=4):
    """Share the same response and Q basis; optionally fix delta_D exactly zero."""
    mapping=nested_hat_transform(objective.basis.shape[1],count)
    transfer=objective.tensor(mapping)
    def response(theta,jacobian):
        parameters=objective.tensor(theta)
        def evaluate(t):
            full=torch.cat((t,t.new_zeros(1))) if physical_dimension==3 else t
            return objective.response(objective.model,objective.sigma,objective.xyt,
                objective.basis,physical_parameters=full,track_gradients=True)
        if jacobian:
            if hasattr(objective.response,'matrix_and_jacobian'):
                full=torch.cat((parameters,parameters.new_zeros(1))) if physical_dimension==3 else parameters
                value,derivative=objective.response.matrix_and_jacobian(objective.model,
                    objective.sigma,objective.xyt,objective.basis,physical_parameters=full)
                derivative=derivative[...,:physical_dimension]
            else:
                with torch.enable_grad():
                    value=evaluate(parameters)
                    derivative=torch.autograd.functional.jacobian(evaluate,parameters,
                        vectorize=True,strategy='forward-mode')
            value=value@transfer
            derivative=torch.einsum('nkd,kj->njd',derivative,transfer)
            return value.detach().cpu().numpy(),derivative.detach().cpu().numpy()
        with torch.no_grad():
            return (evaluate(parameters)@transfer).detach().cpu().numpy()
    return response,mapping


def fit_location_capacity(objective,starts,length_m,counts=(1,3,5),max_nfev=250):
    begun=time.perf_counter()
    bounds=objective.bounds
    cap=bounds.upper[4:]
    if not np.all(cap==cap[0]):
        raise ValueError('Common amplitude bounds required')
    observations=objective.observations.detach().cpu().numpy()
    initial=[np.asarray(start)[:4].copy() for start in starts]
    models=[]
    for count in counts:
        response,mapping=capacity_response(objective,count)
        # Lift prior lower-capacity physical optima into larger nested models.
        fit=fit_reduced_trust_region(response,observations,initial,bounds.lower[:4],bounds.upper[:4],
            q_upper_bound=float(cap[0]),max_nfev=max_nfev)
        theta=fit.pop('theta')
        native=np.asarray(fit['coefficients'])
        value,derivative=response(theta,True)
        physical=np.einsum('nkd,k->nd',derivative,native)
        free=np.asarray(fit['active_mask'])==0
        joint=np.column_stack((physical,value[:,free]))
        fit.update(source_norm=theta[:2],diffusion_norm=[theta[2]+theta[3],theta[2]],
            native_coefficients=native,coefficients=mapping@native,q_basis_count=count,
            mse=fit['sse']/len(observations),parameter_count=4+count,
            # Unit-noise covariance is scaled by the common variance after all fits.
            location_information=location_variance(joint,1.,length_m))
        models.append(fit)
        initial.append(theta.copy())
        print('LOCATION_CAPACITY',count,'done',flush=True)
    selected,variance,fallback=choose_location_capacity(models,len(observations))
    for fit in models:
        info=fit['location_information']
        if info['variance_trace_m2'] is not None:
            info['variance_trace_m2']*=variance
            info['covariance_m2']*=variance
    result=dict(selected)
    result.update(method='ia_location_capacity',seconds=time.perf_counter()-begun,
        selection='Nested-fit compatibility at alpha0.05, then minimum local source variance',
        selection_warning='Gaussian/local/active-set approximation; not coverage-certified under turbulent/model errors',
        noise_variance=variance,weak_information_fallback=fallback,
        release_models=[{key:fit[key] for key in ['q_basis_count','sse','mse','source_norm','diffusion_norm',
            'coefficients','native_coefficients','converged','fit_compatible','fit_excess_sse',
            'fit_compatibility_threshold','location_information']} for fit in models])
    return result
