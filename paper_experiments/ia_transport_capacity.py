"""IA-only nested isotropic/anisotropic diffusion and release-capacity search.

The frozen forward operator is unchanged. Isotropy fixes delta_D=0 by
parameter elimination, not a penalty. Every candidate uses all observations.
"""

import time

import numpy as np

from paper_location_capacity import capacity_response
from q_capacity import gaussian_bic
from reduced_trust_region import fit_reduced_trust_region


def fit_transport_capacity(objective,starts,anisotropic_models,counts=(1,3,5),max_nfev=250):
    begun=time.perf_counter()
    observed=objective.observations.detach().cpu().numpy()
    cap=objective.bounds.upper[4:]
    if not np.all(cap==cap[0]):
        raise ValueError('Common Q coefficient bound required')
    candidates=[]
    scalar_starts=[np.asarray(start)[:3].copy() for start in starts]
    for count in counts:
        response,mapping=capacity_response(objective,count,physical_dimension=3)
        fit=fit_reduced_trust_region(response,observed,scalar_starts,
            objective.bounds.lower[:3],objective.bounds.upper[:3],q_upper_bound=float(cap[0]),max_nfev=max_nfev)
        theta=fit.pop('theta')
        native=np.asarray(fit['coefficients'])
        fit.update(source_norm=theta[:2],diffusion_norm=[theta[2],theta[2]],
            native_coefficients=native,coefficients=mapping@native,q_basis_count=count,
            diffusion_mode='isotropic',parameter_count=3+count,mse=fit['sse']/len(observed))
        candidates.append(fit)
        scalar_starts.append(theta.copy())
        original=next(model for model in anisotropic_models if model['q_basis_count']==count)
        cached=dict(original)
        physical=np.r_[cached['source_norm'],cached['diffusion_norm'][1],
            cached['diffusion_norm'][0]-cached['diffusion_norm'][1]]
        full_response,full_mapping=capacity_response(objective,count)
        prediction=full_response(physical,False)@np.asarray(cached['native_coefficients'])
        sse=float(np.sum((prediction-observed)**2))
        if abs(sse-cached['sse'])>1e-6*max(cached['sse'],1e-30):
            raise ValueError('Cached IA candidate does not match the frozen forward operator')
        cached.update(sse=sse,prediction=prediction,mse=sse/len(observed),
            diffusion_mode='anisotropic',parameter_count=4+count,reused_verified=True)
        # A nested larger model must not lose merely because its old start was poor.
        if fit['sse']<sse-1e-8*max(sse,1e-30):
            initial=[np.asarray(start)[:4] for start in starts]
            initial.extend([physical,np.r_[theta,0.]])
            refined=fit_reduced_trust_region(full_response,observed,initial,
                objective.bounds.lower[:4],objective.bounds.upper[:4],q_upper_bound=float(cap[0]),max_nfev=max_nfev)
            p=refined.pop('theta')
            native=np.asarray(refined['coefficients'])
            refined.update(source_norm=p[:2],diffusion_norm=[p[2]+p[3],p[2]],
                native_coefficients=native,coefficients=full_mapping@native,q_basis_count=count,
                diffusion_mode='anisotropic',parameter_count=4+count,mse=refined['sse']/len(observed),
                refined_from_isotropic=True)
            if refined['sse']<cached['sse']:
                cached=refined
        candidates.append(cached)
        print('TRANSPORT_CAPACITY',count,'done',flush=True)
    for fit in candidates:
        fit['bic']=gaussian_bic(fit['sse'],len(observed),fit['parameter_count'])
    selected=min(candidates,key=lambda model:(model['bic'],model['parameter_count']))
    result=dict(selected)
    result.update(method='ia_transport_capacity',seconds=time.perf_counter()-begun,
        selection='Working Gaussian BIC over nested diffusion and Q capacities; no extra training loss',
        models=[{key:fit[key] for key in ['source_norm','diffusion_norm','coefficients','native_coefficients',
            'q_basis_count','diffusion_mode','parameter_count','sse','mse','bic','converged']} for fit in candidates])
    return result
