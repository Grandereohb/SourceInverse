"""Constant-Q Bayesian puff demo, not Newman M-MALA or a paper reproduction.

The nonnegative scalar rate has a proper half-normal prior and is integrated
out analytically. Only source position and two log diffusivities are sampled.
Fixed wind/quadrature factors are cached; no evaluated truth enters inference.
"""

import numpy as np
from scipy.special import log_ndtr, roots_legendre
from scipy.stats import norm, rankdata
import torch


class ConstantPuffPlan:
    """Cache the independent reference's exact wind integrals and32-order rule."""

    def __init__(self, stations, wind, sigma, samples=16, order=32):
        self.stations=np.asarray(stations,dtype=float)
        if (self.stations.ndim!=2 or self.stations.shape[1]!=2 or
                not np.isfinite(self.stations).all() or sigma<=0 or
                wind not in ('steady','veering') or samples<1 or order<2):
            raise ValueError('Invalid constant puff specification')
        self.sigma=float(sigma)
        self.samples=int(samples)
        times=(np.arange(6*samples)+.5)/samples
        nodes,weights=roots_legendre(order)
        entries=[]
        for end in times:
            breaks=np.r_[np.linspace(0.,6.,5)[np.linspace(0.,6.,5)<end],end]
            left,right=breaks[:-1],breaks[1:]
            release=((left[:,None]+right[:,None])*.5+(right-left)[:,None]*.5*nodes).ravel()
            w=((right-left)[:,None]*.5*weights).ravel()
            age=end-release
            if wind=='steady':
                angle=.05
                c,s=np.cos(angle),np.sin(angle)
                dx,dy=1.2*c*age,1.2*s*age
                pxx,pyy,pxy=c*c*age,s*s*age,c*s*age
            else:
                rate=.15
                start_angle=-.35+rate*release
                angle=-.35+rate*end
                dx=1.2/rate*(np.sin(angle)-np.sin(start_angle))
                dy=1.2/rate*(np.cos(start_angle)-np.cos(angle))
                contrast=(np.sin(2*angle)-np.sin(2*start_angle))/(4*rate)
                pxx,pyy=.5*age+contrast,.5*age-contrast
                pxy=(np.cos(2*start_angle)-np.cos(2*angle))/(4*rate)
            entries.append((age,dx,dy,pxx,pxy,pyy,w*np.exp(-.1*age)))
        width=max(len(v[0]) for v in entries)
        arrays=[np.ones((len(times),width)) for _ in range(7)]
        arrays[-1].fill(0.)
        for i,entry in enumerate(entries):
            for dest,values in zip(arrays,entry):
                dest[i,:len(values)]=values
        self.age,self.dx,self.dy,self.pxx,self.pxy,self.pyy,self.weights=arrays

    def response(self, theta):
        x,y,log_perp,log_delta=np.asarray(theta,dtype=float)
        with np.errstate(over='ignore',invalid='ignore',under='ignore',divide='ignore'):
            dq=.00100005+np.exp(log_perp)
            delta=np.exp(log_delta)
            xx=self.sigma**2+2*(dq*self.age+delta*self.pxx)
            yy=self.sigma**2+2*(dq*self.age+delta*self.pyy)
            xy=2*delta*self.pxy
            determinant=xx*yy-xy*xy
            dx=self.stations[None,:,None,0]-x-self.dx[:,None,:]
            dy=self.stations[None,:,None,1]-y-self.dy[:,None,:]
            exponent=(yy[:,None,:]*dx**2+xx[:,None,:]*dy**2-2*xy[:,None,:]*dx*dy)/determinant[:,None,:]
            kernels=np.exp(-.5*exponent)/(2*np.pi*np.sqrt(determinant[:,None,:]))
            result=np.sum(kernels*self.weights[:,None,:],axis=2)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite Bayesian transport response')
        return result.reshape(6,self.samples,len(self.stations)).mean(1).ravel()


def collapsed_rate(response, observations, noise_std, prior_scale):
    """Log marginal likelihood (up to theta-independent constants), Q|theta N+.

Includes the normalization of the half-normal prior: the zero-response case
has log marginal0 relative to the zero-signal likelihood, for every prior tau.
"""
    g,y=np.asarray(response,dtype=float),np.asarray(observations,dtype=float)
    if (g.ndim!=1 or g.shape!=y.shape or noise_std<=0 or prior_scale<=0 or
            not np.isfinite(g).all() or not np.isfinite(y).all()):
        raise ValueError('Matching finite vectors and positive scales required')
    precision=float(g@g)/(noise_std**2)+1/(prior_scale**2)
    linear=float(g@y)/(noise_std**2)
    mean=linear/precision
    sd=precision**-.5
    z=mean/sd
    log_likelihood=.5*linear*mean-.5*np.log(precision)+log_ndtr(z)+np.log(2/prior_scale)
    return float(log_likelihood),float(mean),float(sd)


def positive_normal_mean(mean, sd):
    mean,sd=np.asarray(mean),np.asarray(sd)
    z=mean/sd
    return mean+sd*np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z))


class CollapsedPosterior:
    def __init__(self, plan, observed, noise_std=.01, prior_scale=5.):
        self.plan=plan
        self.observed=np.asarray(observed,dtype=float)
        self.noise_std,self.prior_scale=float(noise_std),float(prior_scale)
        self.calls=0

    def __call__(self, theta):
        self.calls+=1
        theta=np.asarray(theta,dtype=float)
        if (theta.shape!=(4,) or not np.isfinite(theta).all() or np.any(np.abs(theta[:2])>.75)):
            return -np.inf,np.nan,np.nan
        # Priors are on log(D_perp-floor) and log(D_parallel-D_perp).
        # No extra Jacobian: these are the sampled coordinates with Normal priors.
        log_prior=-.5*np.sum(((theta[2:]-np.log([.02,.03]))/1.5)**2)
        if not np.isfinite(log_prior) or log_prior < -1e6:
            return -np.inf,np.nan,np.nan
        try:
            value,mean,sd=collapsed_rate(self.plan.response(theta),self.observed,
                                       self.noise_std,self.prior_scale)
        except FloatingPointError:
            return -np.inf,np.nan,np.nan
        return value+log_prior,mean,sd


class BatchedPosterior:
    """Same target in float64 batches; emcee still supplies the sampling engine."""

    def __init__(self, posterior, device='cpu'):
        self.original=posterior
        self.device=torch.device(device)
        p=posterior.plan
        cast=lambda v:torch.as_tensor(v,dtype=torch.float64,device=device)
        self.age,self.dx,self.dy,self.pxx,self.pxy,self.pyy,self.weights=[cast(v) for v in
            (p.age,p.dx,p.dy,p.pxx,p.pxy,p.pyy,p.weights)]
        self.stations=cast(p.stations)
        self.observed=cast(posterior.observed)
        self.prior_center=cast(np.log([.02,.03]))
        self.calls=0

    def __call__(self, theta):
        a=torch.as_tensor(np.asarray(theta),dtype=torch.float64,device=self.device)
        if a.ndim!=2 or a.shape[1]!=4:
            raise ValueError('Expected batch of four-parameter coordinates')
        self.calls+=len(a)
        valid=torch.isfinite(a).all(1)&(a[:,:2].abs()<=.75).all(1)
        output=torch.full((len(a),3),float('nan'),dtype=a.dtype,device=a.device)
        output[:,0]=float('-inf')
        if bool(valid.any()):
            v=a[valid]
            dq=.00100005+v[:,2,None,None].exp()
            delta=v[:,3,None,None].exp()
            xx=self.original.plan.sigma**2+2*(dq*self.age+delta*self.pxx)
            yy=self.original.plan.sigma**2+2*(dq*self.age+delta*self.pyy)
            xy=2*delta*self.pxy
            det=xx*yy-xy*xy
            dx=self.stations[None,None,:,None,0]-v[:,None,None,None,0]-self.dx[None,:,None,:]
            dy=self.stations[None,None,:,None,1]-v[:,None,None,None,1]-self.dy[None,:,None,:]
            exponent=(yy[:,:,None,:]*dx**2+xx[:,:,None,:]*dy**2-2*xy[:,:,None,:]*dx*dy)/det[:,:,None,:]
            response=(torch.exp(-.5*exponent)/(2*np.pi*det[:,:,None,:].sqrt())
                      *self.weights[None,:,None,:]).sum(-1)
            response=response.reshape(len(v),6,self.original.plan.samples,len(self.stations)).mean(2).flatten(1)
            precision=response.square().sum(1)/self.original.noise_std**2+1/self.original.prior_scale**2
            linear=(response*self.observed).sum(1)/self.original.noise_std**2
            mean=linear/precision
            sd=precision.rsqrt()
            value=.5*linear*mean-.5*precision.log()+torch.special.log_ndtr(mean/sd)+np.log(2/self.original.prior_scale)
            prior=-.5*((v[:,2:]-self.prior_center)/1.5).square().sum(1)
            output[valid]=torch.stack((value+prior,mean,sd),1)
        rows=output.cpu().numpy()
        rows[~np.isfinite(rows[:,0]),0]=-np.inf
        return [tuple(row) for row in rows]


def rank_split_rhat(chains):
    """Rank and folded split-Rhat for INDEPENDENT chains, never ensemble walkers."""
    x=np.asarray(chains,dtype=float)
    if x.ndim!=3 or x.shape[0]<2 or x.shape[1]<8 or not np.isfinite(x).all():
        raise ValueError('Finite independent chain, draw, parameter array required')
    half=x.shape[1]//2
    split=np.concatenate((x[:,:half],x[:,-half:]),axis=0)
    results=[]
    for index in range(x.shape[2]):
        values=split[:,:,index]
        estimates=[]
        for v in (values,np.abs(values-np.median(values))):
            ranks=rankdata(v.ravel(),method='average')
            ranked=norm.ppf((ranks-.375)/(len(ranks)+.25)).reshape(v.shape)
            within=np.mean(np.var(ranked,axis=1,ddof=1))
            between=half*np.var(np.mean(ranked,axis=1),ddof=1)
            estimates.append(float(np.sqrt(((half-1)*within+between)/(half*within)))
                             if within>0 else np.inf)
        results.append(max(estimates))
    return np.asarray(results)
