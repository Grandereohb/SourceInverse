"""Nonnegative release-support model selection on one shared transport dictionary.

Each candidate uses the original squared observation residual and eliminates Q.
Working Gaussian BIC chooses among uniformly weighted discrete models; there is
no tuned sparsity penalty, source truth, or observation withholding.
"""

from itertools import combinations
import time

import numpy as np
import torch

from paper_joint_baselines import JointBounds,JointResidual
from paper_reduced_solver import fit_profiled_trf
from q_capacity import nested_hat_transform,gaussian_bic
from reduced_trust_region import ReducedResidual


def release_support_models(base_count=5):
    if base_count<3 or base_count>7 or base_count%2==0:
        raise ValueError('An odd shared dictionary of3-7 hats is supported')
    models=[('constant',np.ones((base_count,1))),
        ('nested_three',nested_hat_transform(base_count,3))]
    for size in range(1,base_count+1):
        for indices in combinations(range(base_count),size):
            models.append(('support_'+'_'.join(map(str,indices)),np.eye(base_count)[:,indices]))
    return models


class MappedReleaseResponse:
    def __init__(self,original,full_basis,mapping):
        self.original,self.full_basis,self.mapping=original,full_basis,mapping
        if hasattr(original,'matrix_and_jacobian'):
            self.matrix_and_jacobian=self._matrix_and_jacobian

    def __call__(self,model,sigma,xyt,basis,**kwargs):
        return self.original(model,sigma,xyt,self.full_basis,**kwargs)@self.mapping

    def _matrix_and_jacobian(self,model,sigma,xyt,basis,**kwargs):
        value,derivative=self.original.matrix_and_jacobian(model,sigma,xyt,self.full_basis,**kwargs)
        return value@self.mapping,torch.einsum('nkd,kj->njd',derivative,self.mapping)


def fit_release_support(objective,starts,max_nfev=100,refine_all=False):
    begun=time.perf_counter()
    cap=objective.bounds.upper[4:]
    if not np.all(cap==cap[0]):
        raise ValueError('Shared physical amplitude bound is required')
    definitions=release_support_models(objective.basis.shape[1])
    full_id=definitions[-1][0]
    full=fit_profiled_trf(objective,starts,max_nfev)
    theta=np.r_[full['source_norm'],full['diffusion_norm'][1],
        full['diffusion_norm'][0]-full['diffusion_norm'][1]]
    warm=np.r_[theta,np.zeros(objective.basis.shape[1])]
    working_starts=[*starts,warm]
    screening=[]
    selected={identifier for identifier,_ in definitions} if refine_all else {'constant','nested_three',full_id}
    if not refine_all:
        matrices=[objective.response(objective.model,objective.sigma,objective.xyt,objective.basis,
            physical_parameters=objective.tensor(s[:4])).detach().cpu().numpy() for s in working_starts]
        observed=objective.observations.detach().cpu().numpy()
        for identifier,mapping in definitions:
            scores=[]
            for matrix in matrices:
                reduced=ReducedResidual(lambda unused,j,m=matrix@mapping:m,observed,
                    q_upper_bound=float(cap[0]))
                residual=reduced.residual(np.zeros(4))
                scores.append(gaussian_bic(float(residual@residual),len(observed),4+mapping.shape[1]))
            screening.append(dict(layout_id=identifier,count=mapping.shape[1],bic=min(scores)))
        for size in range(1,objective.basis.shape[1]):
            pool=[s for s in screening if s['layout_id'].startswith('support_') and s['count']==size]
            selected.add(min(pool,key=lambda item:(item['bic'],item['layout_id']))['layout_id'])
    candidates=[]
    for identifier,mapping in definitions:
        if identifier not in selected:
            continue
        transfer=objective.tensor(mapping)
        small_basis=objective.tensor(objective.basis)@transfer
        wrapped=MappedReleaseResponse(objective.response,objective.basis,transfer)
        count=mapping.shape[1]
        bounds=JointBounds(np.r_[objective.bounds.lower[:4],np.zeros(count)],
            np.r_[objective.bounds.upper[:4],np.full(count,cap[0])])
        small=JointResidual(wrapped,objective.model,objective.sigma,objective.xyt,
            small_basis,objective.observations,bounds)
        fit=(dict(full) if identifier==full_id else
            fit_profiled_trf(small,[np.r_[s[:4],np.zeros(count)] for s in working_starts],max_nfev))
        native=np.asarray(fit['coefficients'])
        fit.update(layout_id=identifier,native_coefficients=native.tolist(),
            coefficients=(mapping@native).tolist(),q_basis_count=count,
            bic=gaussian_bic(fit['sse'],len(fit['prediction']),4+count),
            model_prior='Uniform over the declared discrete model list; BIC working likelihood')
        candidates.append(fit)
        print('SUPPORT_LAYOUT',identifier,'done',flush=True)
    selected=min(candidates,key=lambda item:(item['bic'],item['q_basis_count'],item['layout_id']))
    result=dict(selected)
    result.update(method='ia_vpdi_release_support',seconds=time.perf_counter()-begun,
        screening=screening,refined_layouts=[fit['layout_id'] for fit in candidates],candidate_layout_count=len(definitions),
        release_models=[{key:fit[key] for key in ['layout_id','bic','sse','source_norm',
            'diffusion_norm','coefficients','native_coefficients','q_basis_count','converged','seconds']}
            for fit in candidates])
    return result
