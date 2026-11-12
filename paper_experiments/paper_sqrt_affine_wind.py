"""Square-root Gaussian covariance transport; no diffusivity/eigenvalue clipping."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache

from paper_affine_wind import AffineWindHistory,affine_step,traceless_coefficients


def covariance_factor(columns):
    """Return L with L L.T=columns columns.T via stable thin QR."""
    return np.linalg.qr(np.swapaxes(columns,-1,-2),mode='r').swapaxes(-1,-2)


@lru_cache(maxsize=4)
def _quadrature(order):
    return leggauss(order)


def step_factors(gradient,velocity,duration,direction,order=16):
    """Positive quadrature reference, independent of the closed-form factor."""
    flow,translation,_=affine_step(gradient,velocity,duration,np.eye(2))
    nodes,weights=_quadrature(order)
    duration=np.asarray(duration,float)
    clock=duration[...,None]*(nodes+1)/2
    transforms,_,_=affine_step(gradient,velocity,clock,np.eye(2))
    root_weight=np.sqrt(duration[...,None]*weights/2)
    perpendicular=np.array([-direction[1],direction[0]])
    def integrate(axis):
        columns=np.einsum('...nij,j->...in',transforms,axis)*root_weight[...,None,:]
        return covariance_factor(columns)
    return flow,translation,integrate(direction),integrate(perpendicular)


def exact_step_factors(gradient,velocity,duration,direction):
    a,b=np.asarray(gradient,float),np.asarray(velocity,float)
    t=np.asarray(duration,float)
    gamma=.5*np.trace(a@a)
    z=gamma*t*t
    cosine,sine,integrated_sine,iss=traceless_coefficients(a,t)
    icc=t+gamma*iss
    ics=sine*sine/2
    determinant=t**4*(1/12+z/90+z*z/1260+z*z*z/28350)
    if gamma!=0:
        determinant=np.where(np.abs(z)<1e-3,determinant,(sine*sine-t*t)/(4*gamma))
    root=np.sqrt(icc)
    denominator=np.where(t>0,root,1.)
    lower=np.stack((np.stack((root,np.zeros_like(root)),axis=-1),
        np.stack((ics/denominator,np.sqrt(determinant/np.where(t>0,icc,1.))),axis=-1)),axis=-2)
    def integrate(axis):
        return np.column_stack((axis,a@axis))@lower
    flow=cosine[...,None,None]*np.eye(2)+sine[...,None,None]*a
    translation=sine[...,None]*b+integrated_sine[...,None]*(a@b)
    return flow,translation,integrate(direction),integrate(np.array([-direction[1],direction[0]]))


class SquareRootAffineWindHistory(AffineWindHistory):
    def __init__(self,edges,velocity,gradient,diffusion_direction):
        super().__init__(edges,velocity,gradient,diffusion_direction)
        wind=np.asarray(diffusion_direction,float)
        speed=np.linalg.norm(wind,axis=1)
        self.direction=wind/np.maximum(speed[:,None],np.finfo(float).tiny)
        self.direction[speed==0]=[1.,0.]

    def relative_square_root(self,end,releases):
        releases=np.asarray(releases,float)
        maps=np.zeros((len(releases),2,2))
        displacement=np.zeros((len(releases),2))
        par=np.zeros_like(maps)
        per=np.zeros_like(maps)
        front=np.eye(2)
        shift=np.zeros(2)
        p_front=np.zeros((2,2))
        q_front=np.zeros((2,2))
        last=np.searchsorted(self.edges,end,side='left')-1
        first=max(0,np.searchsorted(self.edges,releases.min(),side='right')-1)
        for index in range(last,first-1,-1):
            upper=min(end,self.edges[index+1])
            mask=(releases>=self.edges[index])&(releases<=upper)
            times=np.r_[upper-releases[mask],upper-self.edges[index]]
            flow,translation,p_step,q_step=exact_step_factors(self.gradient[index],self.velocity[index],
                times,self.direction[index])
            transported=front@flow
            shifted=shift+np.einsum('ij,nj->ni',front,translation)
            p=covariance_factor(np.concatenate((np.broadcast_to(p_front,p_step.shape),front@p_step),axis=-1))
            q=covariance_factor(np.concatenate((np.broadcast_to(q_front,q_step.shape),front@q_step),axis=-1))
            maps[mask],displacement[mask],par[mask],per[mask]=transported[:-1],shifted[:-1],p[:-1],q[:-1]
            front,shift,p_front,q_front=transported[-1],shifted[-1],p[-1],q[-1]
        if not all(np.isfinite(value).all() for value in (maps,displacement,par,per)):
            raise FloatingPointError('Square-root affine flow overflowed; no physical clipping applied')
        return maps,displacement,par,per

    def relative(self,end,releases):
        maps,displacement,lp,lq=self.relative_square_root(end,releases)
        p,q=lp@lp.swapaxes(-1,-2),lq@lq.swapaxes(-1,-2)
        return maps,displacement,p[:,[0,0,1],[0,1,1]],q[:,[0,0,1],[0,1,1]]
