"""IA-only low-dimensional wind calibration independent of concentration.

At minute t, u(x,t)=mean_u(t)+A(t)(x-center), where
A(t)=mean_u_x(t) Bx+mean_u_y(t) By and tr(Bx)=tr(By)=0.
Six shared spatial coefficients replace three free gradients every minute.
This is a local linear flow approximation, not a CFD or turbulence model.
"""

import numpy as np


def fit_shared_wind_transfer(positions,velocities):
    positions=np.asarray(positions,float)
    velocities=np.asarray(velocities,float)
    if (positions.ndim!=2 or positions.shape[1]!=2 or len(positions)<3
            or velocities.ndim!=3 or velocities.shape[1:]!=positions.shape
            or not len(velocities) or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
            or np.linalg.matrix_rank(positions-positions.mean(0))<2):
        raise ValueError('Finite noncollinear wind stations and matching vector history required')
    center=positions.mean(0)
    relative=positions-center
    mean=velocities.mean(1)
    x,y=relative.T
    u,v=mean.T
    design=np.zeros((*velocities.shape,6))
    design[:,:,0,0]=u[:,None]*x
    design[:,:,0,1]=u[:,None]*y
    design[:,:,0,3]=v[:,None]*x
    design[:,:,0,4]=v[:,None]*y
    design[:,:,1,0]=-u[:,None]*y
    design[:,:,1,2]=u[:,None]*x
    design[:,:,1,3]=-v[:,None]*y
    design[:,:,1,5]=v[:,None]*x
    target=(velocities-mean[:,None,:]).reshape(-1)
    matrix=design.reshape(-1,6)
    coefficients,_,rank,singular=np.linalg.lstsq(matrix,target,rcond=None)
    bx=np.array([[coefficients[0],coefficients[1]],[coefficients[2],-coefficients[0]]])
    by=np.array([[coefficients[3],coefficients[4]],[coefficients[5],-coefficients[3]]])
    gradient=u[:,None,None]*bx+v[:,None,None]*by
    intercept=mean-np.einsum('tij,j->ti',gradient,center)
    fitted=intercept[:,None,:]+np.einsum('tij,sj->tsi',gradient,positions)
    sse=float(np.sum((fitted-velocities)**2))
    uniform_sse=float(target@target)
    return dict(velocity=intercept,gradient=gradient,fitted=fitted,
        station_center_m=center,bx_per_m=bx,by_per_m=by,
        shared_parameter_rank=int(rank),shared_parameter_count=6,
        measured_wind_component_rank=int(np.linalg.matrix_rank(mean)),
        design_condition=float(singular[0]/singular[rank-1]) if rank else None,
        wind_sse=sse,uniform_wind_sse=uniform_sse,
        variance_fraction_explained=1-sse/uniform_sse if uniform_sse>0 else None,
        note='All supplied wind vectors; no concentration or source truth; do not interpret Eulerian minute variability as diffusivity')
