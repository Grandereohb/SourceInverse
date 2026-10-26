"""Classical conditional-amplitude stationary AD Green-function demo.

Measured wind is frozen separately for each minute, with an infinite stationary
release history. Same3D source/ground/ppm assumptions as the physical adapter,
but no time-varying Q or historical wind transport. Not a named-paper replica.
"""

import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import log_ndtr

from paper_physical_beam_adapter import ppm_per_kg_m3


def steady_beam_response(wind,starts,ends,source,height,d_parallel,d_perp,temperature,pressure,
                         *,order=128,source_variance=1/3):
    wind,starts,ends,source=[np.asarray(v,dtype=float) for v in (wind,starts,ends,source)]
    if (wind.ndim!=2 or wind.shape[1]!=2 or starts.ndim!=2 or starts.shape[1]!=3
            or starts.shape!=ends.shape or source.shape!=(2,)
            or not np.all(starts[:,2]==ends[:,2]) or not d_parallel>=d_perp>0
            or source_variance<0 or height<0 or
            not all(np.isfinite(v).all() for v in (wind,starts,ends,source))):
        raise ValueError('Invalid stationary physical beam specification')
    speed=np.linalg.norm(wind,axis=1)
    direction=wind/np.maximum(speed[:,None],np.finfo(float).tiny)
    direction[speed==0]=[1.,0.]
    nodes,weights=leggauss(order)
    root_scale=np.sqrt(100./np.maximum(speed,.1))[:,None]
    roots=root_scale*(1+nodes)/(1-nodes)
    age=roots**2
    jacobian=4*root_scale*roots/(1-nodes)**2
    time_weights=jacobian*weights
    contrast=d_parallel-d_perp
    xx=source_variance+2*age*(d_perp+contrast*direction[:,0,None]**2)
    yy=source_variance+2*age*(d_perp+contrast*direction[:,1,None]**2)
    xy=2*age*contrast*direction[:,0,None]*direction[:,1,None]
    dx=starts[None,:,None,0]-source[0]-wind[:,None,0,None]*age[:,None,:]
    dy=starts[None,:,None,1]-source[1]-wind[:,None,1,None]*age[:,None,:]
    lx,ly=(ends-starts)[None,:,None,:2].transpose(3,0,1,2)
    determinant=xx*yy-xy*xy
    ad=yy[:,None,:]*lx**2+xx[:,None,:]*ly**2-2*xy[:,None,:]*lx*ly
    bn=yy[:,None,:]*lx*dx+xx[:,None,:]*ly*dy-xy[:,None,:]*(lx*dy+ly*dx)
    denominator=np.sqrt(ad*determinant[:,None,:])
    lower,upper=bn/denominator,(ad+bn)/denominator
    lo,hi=np.where(lower>=0,-upper,lower),np.where(lower>=0,-lower,upper)
    log_lo,log_hi=log_ndtr(lo),log_ndtr(hi)
    horizontal=np.exp(-.5*(lx*dy-ly*dx)**2/ad)*np.exp(log_hi)*(-np.expm1(log_lo-log_hi))/np.sqrt(2*np.pi*ad)
    variance=2*d_perp*age[:,None,:]
    z=starts[None,:,None,2]
    vertical=(np.exp(-.5*(z-height)**2/variance)+np.exp(-.5*(z+height)**2/variance))/np.sqrt(2*np.pi*variance)
    result=np.sum(horizontal*vertical*time_weights[:,None,:],axis=2)
    return (result*np.asarray(ppm_per_kg_m3(temperature,pressure))[:,None]).ravel()


def fit_steady_baseline(wind,starts,ends,height,temperature,pressure,observed,
                        *,max_evaluations=3000,order=128):
    observed=np.asarray(observed,dtype=float)
    best=None
    calls=0
    def evaluate(theta):
        nonlocal best,calls
        calls+=1
        with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
            dq=1/6+np.exp(theta[2])
            dp=dq+np.exp(theta[3])
            if not np.isfinite([dp,dq]).all():
                return np.inf
            response=steady_beam_response(wind,starts,ends,np.asarray(theta[:2])*100,height,
                dp,dq,temperature,pressure,order=order)
        if not np.isfinite(response).all():
            return np.inf
        norm=float(response@response)
        amplitude=max(float(response@observed)/norm,0.) if norm>0 else 0.
        prediction=response*amplitude
        mse=float(np.mean((prediction-observed)**2))
        if best is None or mse<best['observed_mse']:
            best=dict(source_norm=np.asarray(theta[:2]).tolist(),diffusion_m2_s=[float(dp),float(dq)],
                q_kg_s=amplitude,prediction_ppm=prediction.tolist(),observed_mse=mse)
        return mse
    started=time.perf_counter()
    coarse=[]
    for scale in (.5,1.,2.):
        for ratio in (1.,3.):
            dq=6.5*scale
            for x in np.linspace(0.,1.1,7):
                for y in np.linspace(0.,1.1,7):
                    theta=np.array([x,y,np.log(dq-1/6),np.log(max(dq*(ratio-1),1e-6))])
                    coarse.append((evaluate(theta),theta))
    coarse.sort(key=lambda item:item[0])
    first=coarse[0][1]
    second=next(v for _,v in coarse if np.linalg.norm(v[:2]-first[:2])>=.2)
    terminations=[]
    for start in (first,second):
        run=minimize(evaluate,start,method='Powell',bounds=[(0.,1.1),(0.,1.1),(None,None),(None,None)],
            options=dict(maxfev=max_evaluations,maxiter=30,xtol=1e-4,ftol=1e-8))
        terminations.append(dict(success=bool(run.success),message=str(run.message),nfev=int(run.nfev)))
    fine=steady_beam_response(wind,starts,ends,np.asarray(best['source_norm'])*100,height,
        *best['diffusion_m2_s'],temperature,pressure,order=2*order)*best['q_kg_s']
    difference=float(np.linalg.norm(fine-best['prediction_ppm'])/max(np.linalg.norm(fine),np.finfo(float).tiny))
    return dict(**best,seconds=time.perf_counter()-started,response_calls=calls,
        terminations=terminations,converged=all(t['success'] for t in terminations),
        quadrature_relative_replay_difference=difference,quadrature_gate_passed=difference<=.01,
        scope='stationary wind-per-minute conditional-amplitude3D AD demo; not a named-paper reproduction')
