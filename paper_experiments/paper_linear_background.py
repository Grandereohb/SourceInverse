"""Joint linear elimination of Q and a shared, slowly varying ambient background.

The source-only physical response is unchanged. Independent pre-event samples
anchor station offsets and initial ambient level; every event observation is used.
"""

import numpy as np

from reduced_trust_region import fit_reduced_trust_region
from q_capacity import nested_hat_transform,gaussian_bic


def background_design(shape,finite_indices,pre_event,count=3):
    duration,stations=shape
    pre=np.asarray(pre_event,dtype=float)
    if pre.ndim!=2 or pre.shape[1]!=stations or count<2:
        raise ValueError('Compatible independent background table required')
    medians=np.nanmedian(pre,axis=0)
    if not np.isfinite(medians).all():
        raise ValueError('Each station needs independently observed background')
    offsets=medians-np.median(medians)
    indices=np.asarray(finite_indices,dtype=int)
    knots=np.linspace(0.,float(duration),count)
    time=indices//stations+.5
    event_basis=np.column_stack([np.interp(time,knots,np.eye(count)[j]) for j in range(count)])
    pre_mask=np.isfinite(pre)
    pre_values=(pre-offsets)[pre_mask]
    calibration=np.zeros((len(pre_values),count))
    calibration[:,0]=1.
    return dict(event_basis=event_basis,calibration_basis=calibration,
        pre_values=pre_values,offsets=offsets,event_offsets=offsets[indices%stations],knots=knots)


def fit_source_and_background(objective,starts,design,counts=(1,3,5),max_nfev=250):
    source_response=objective.response
    n_event=objective.observations.numel()
    event_observed=objective.observations.detach().cpu().numpy()-design['event_offsets']
    observed=np.r_[event_observed,design['pre_values']]
    n_pre=len(design['pre_values'])
    bg=np.vstack((design['event_basis'],design['calibration_basis']))
    bg_count=bg.shape[1]
    models=[]
    for count in counts:
        transform=nested_hat_transform(objective.basis.shape[1],count)
        transfer=objective.tensor(transform)
        def matrix(values,jacobian):
            theta=objective.tensor(values)
            if jacobian and hasattr(source_response,'matrix_and_jacobian'):
                value,derivative=source_response.matrix_and_jacobian(objective.model,
                    objective.sigma,objective.xyt,objective.basis,physical_parameters=theta)
                value=value@transfer
                import torch
                derivative=torch.einsum('nkd,kj->njd',derivative,transfer)
            else:
                def calculate(parameters):
                    return source_response(objective.model,objective.sigma,objective.xyt,
                        objective.basis,track_gradients=jacobian,physical_parameters=parameters)@transfer
                value=calculate(theta)
                if jacobian:
                    import torch
                    derivative=torch.autograd.functional.jacobian(calculate,theta,
                        vectorize=True,strategy='forward-mode')
            value=value.detach().cpu().numpy()
            source=np.vstack((value,np.zeros((n_pre,count))))
            augmented=np.column_stack((source,bg))
            if not jacobian:
                return augmented
            gradient=np.concatenate((derivative.detach().cpu().numpy(),np.zeros((n_pre,count,4))),axis=0)
            return augmented,np.concatenate((gradient,np.zeros((len(observed),bg_count,4))),axis=1)
        fit=fit_reduced_trust_region(matrix,observed,[s[:4] for s in starts],
            objective.bounds.lower[:4],objective.bounds.upper[:4],
            q_upper_bound=np.r_[np.full(count,objective.bounds.upper[4]),np.full(bg_count,np.inf)],
            max_nfev=max_nfev)
        theta=fit.pop('theta')
        native=fit.pop('coefficients')
        adjusted_prediction=np.asarray(fit.pop('prediction'))
        fit.update(source_norm=theta[:2].tolist(),diffusion_norm=[theta[2]+theta[3],theta[2]],
            coefficients=(transform@native[:count]).tolist(),native_coefficients=native[:count].tolist(),
            background_coefficients=native[count:].tolist(),q_basis_count=count,
            prediction=(adjusted_prediction[:n_event]+design['event_offsets']).tolist(),
            source_prediction=(adjusted_prediction[:n_event]-design['event_basis']@native[count:]).tolist(),
            bic=gaussian_bic(fit['sse'],len(observed),4+count+bg_count+len(design['offsets'])-1),
            method='ia_vpdi_joint_linear_background',background_knots_minutes=design['knots'].tolist())
        raw=objective.observations.detach().cpu().numpy()
        fit['mse']=float(np.mean((np.asarray(fit['prediction'])-raw)**2))
        models.append(fit)
    selected=min(models,key=lambda item:(item['bic'],item['q_basis_count']))
    result=dict(selected)
    result['release_models']=[{key:fit[key] for key in
        ['q_basis_count','bic','sse','source_norm','diffusion_norm','coefficients','converged','seconds']} for fit in models]
    return result
