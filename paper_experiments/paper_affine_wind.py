"""Exact Gaussian transport in a prescribed piecewise-affine horizontal wind.

The velocity is b(t)+A(t)x with tr(A)=0. A and b are independently fitted
to wind sensors, not concentration or source truth. Horizontal covariance
obeys dC/dt=A C+C A.T+2K. Q remains an external linear amplitude.
"""

import numpy as np


def fit_incompressible_wind(positions,velocities):
    positions,velocities=np.asarray(positions,float),np.asarray(velocities,float)
    if positions.ndim!=2 or positions.shape[1]!=2 or velocities.shape[1:]!=positions.shape:
        raise ValueError('Wind histories must be time by sensor by2')
    design=np.array([row for x,y in positions for row in ([1,0,x,y,0],[0,1,-y,0,x])],float)
    if np.linalg.matrix_rank(design)<5 or not np.isfinite(velocities).all():
        raise ValueError('At least three noncollinear finite wind observations required')
    coefficients=velocities.reshape(len(velocities),-1)@np.linalg.pinv(design).T
    gradient=np.zeros((len(velocities),2,2))
    gradient[:,0,0]=coefficients[:,2]
    gradient[:,1,1]=-coefficients[:,2]
    gradient[:,0,1]=coefficients[:,3]
    gradient[:,1,0]=coefficients[:,4]
    fitted=(coefficients@design.T).reshape(velocities.shape)
    return coefficients[:,:2],gradient,fitted


def traceless_coefficients(gradient,duration):
    """Stable scalar coefficients of exp(A t), A squared=gamma I."""
    a=np.asarray(gradient,float)
    t=np.asarray(duration,float)
    gamma=.5*np.trace(a@a)
    z=gamma*t*t
    if gamma==0:
        cosine=np.ones_like(t)
        sine=t.copy()
    elif gamma>0:
        root=np.sqrt(gamma)
        cosine=np.cosh(root*t)
        sine=np.sinh(root*t)/root
    else:
        root=np.sqrt(-gamma)
        cosine=np.cos(root*t)
        sine=np.sin(root*t)/root
    small=np.abs(z)<1e-3
    # Taylor limits avoid cancellation for shear, calm and short intervals.
    integrated_sine=t*t*(.5+z/24+z*z/720+z*z*z/40320)
    iss=t**3*(1/3+z/15+2*z*z/315+z*z*z/2835)
    if gamma!=0:
        integrated_sine=np.where(small,integrated_sine,(cosine-1)/gamma)
        if gamma>0:
            raw=(np.sinh(2*np.sqrt(gamma)*t)/(4*np.sqrt(gamma))-t/2)/gamma
        else:
            raw=(np.sin(2*np.sqrt(-gamma)*t)/(4*np.sqrt(-gamma))-t/2)/gamma
        iss=np.where(small,iss,raw)
    return cosine,sine,integrated_sine,iss


def affine_step(gradient,velocity,duration,tensor):
    """Exact flow/translation and integral E(s) K E(s).T for traceless2x2 A."""
    a,b,k=np.asarray(gradient,float),np.asarray(velocity,float),np.asarray(tensor,float)
    t=np.asarray(duration,float)
    gamma=.5*np.trace(a@a)
    cosine,sine,integrated_sine,iss=traceless_coefficients(a,t)
    identity=np.eye(2)
    flow=cosine[...,None,None]*identity+sine[...,None,None]*a
    translation=sine[...,None]*b+integrated_sine[...,None]*(a@b)
    covariance=((t+gamma*iss)[...,None,None]*k+
        (sine*sine/2)[...,None,None]*(a@k+k@a.T)+
        iss[...,None,None]*(a@k@a.T))
    return flow,translation,covariance


class AffineWindHistory:
    def __init__(self,edges,velocity,gradient,diffusion_direction):
        self.edges=np.asarray(edges,float)
        self.velocity=np.asarray(velocity,float)
        self.gradient=np.asarray(gradient,float)
        wind=np.asarray(diffusion_direction,float)
        speed=np.linalg.norm(wind,axis=1)
        direction=wind/np.maximum(speed[:,None],np.finfo(float).tiny)
        direction[speed==0]=[1.,0.]
        self.parallel=direction[:,:,None]*direction[:,None,:]
        if (self.velocity.shape!=(len(self.edges)-1,2) or self.gradient.shape!=(len(self.velocity),2,2)
                or not np.isfinite(self.gradient).all() or
                not np.allclose(np.trace(self.gradient,axis1=1,axis2=2),0.,atol=1e-12)):
            raise ValueError('Matching finite traceless wind history required')

    def relative(self,end,releases):
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
            flow,translation,p_step=affine_step(self.gradient[index],self.velocity[index],
                times,self.parallel[index])
            _,_,q_step=affine_step(self.gradient[index],self.velocity[index],
                times,np.eye(2)-self.parallel[index])
            transported=front@flow
            shifted=shift+np.einsum('ij,nj->ni',front,translation)
            p=p_front+front@p_step@front.T
            q=q_front+front@q_step@front.T
            maps[mask],displacement[mask],par[mask],per[mask]=transported[:-1],shifted[:-1],p[:-1],q[:-1]
            front,shift,p_front,q_front=transported[-1],shifted[-1],p[-1],q[-1]
        if not all(np.isfinite(value).all() for value in (maps,displacement,par,per)):
            raise FloatingPointError('Affine flow cannot be represented at this history length')
        return maps,displacement,par[:,[0,0,1],[0,1,1]],per[:,[0,0,1],[0,1,1]]
